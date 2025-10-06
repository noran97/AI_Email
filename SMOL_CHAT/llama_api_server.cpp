// llama_api_server_persona.cpp
#include "httplib.h"
#include "llama.h"
#include "common.h"
#include "json.hpp"
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cstring>

using json = nlohmann::json;

class LlamaInference {
private:
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    llama_context_params ctx_params{};
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler_state{nullptr, llama_sampler_free};

public:
    LlamaInference(const std::string& model_path) {
        llama_backend_init();

        llama_model_params mparams = llama_model_default_params();
        model = llama_model_load_from_file(model_path.c_str(), mparams);
        if (!model) throw std::runtime_error("Failed to load model from: " + model_path);

        ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 2048;
        ctx_params.n_threads = 4;
        ctx_params.n_batch = 512;
        ctx = llama_init_from_model(model, ctx_params);
        if (!ctx) {
            llama_model_free(model);
            throw std::runtime_error("Failed to create context");
        }

        llama_sampler_chain_params schain_params = llama_sampler_chain_default_params();
        sampler_state.reset(llama_sampler_chain_init(schain_params));
        if (!sampler_state) {
            llama_free(ctx);
            llama_model_free(model);
            throw std::runtime_error("Failed to initialize sampler chain");
        }

        llama_sampler_chain_add(sampler_state.get(), llama_sampler_init_top_k(40));
        llama_sampler_chain_add(sampler_state.get(), llama_sampler_init_top_p(0.9f, 1));
        llama_sampler_chain_add(sampler_state.get(), llama_sampler_init_temp(0.7f));
        llama_sampler_chain_add(sampler_state.get(), llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }

    ~LlamaInference() {
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
        llama_backend_free();
    }

    void recreate_context_for_fresh_generation() {
        if (ctx) {
            llama_free(ctx);
            ctx = nullptr;
        }
        ctx = llama_init_from_model(model, ctx_params);
        if (!ctx) throw std::runtime_error("Failed to recreate context");
    }

    std::string generate(const std::string& prompt, int max_tokens = 512) {
        if (!model) throw std::runtime_error("Model not loaded");
        if (!ctx) ctx = llama_init_from_model(model, ctx_params);

        const llama_model* model_info = llama_get_model(ctx);
        const llama_vocab* vocab = llama_model_get_vocab(model_info);

        // Tokenize prompt
        std::vector<llama_token> tokens;
        tokens.resize(prompt.size() * 4 + 16);

        int n_tokens = llama_tokenize(vocab,
                                     prompt.c_str(), (int)prompt.size(),
                                     tokens.data(), (int)tokens.size(),
                                     true,
                                     false);
        if (n_tokens < 0) {
            throw std::runtime_error("Tokenization failed");
        }
        tokens.resize(n_tokens);

        // Clear sampler state
        llama_sampler_reset(sampler_state.get());

        // Prepare batch for prompt
        llama_batch batch = llama_batch_init(n_tokens, 0, 1);
        batch.n_tokens = n_tokens;
        
        for (int i = 0; i < n_tokens; ++i) {
            batch.token[i]   = tokens[i];
            batch.pos[i]     = i;
            batch.logits[i]  = (i == n_tokens - 1);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
        }

        llama_memory_clear(llama_get_memory(ctx), false);
        
        // Decode prompt
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            throw std::runtime_error("Failed to decode prompt");
        }

        llama_batch_free(batch);

        // Make sampler aware of prompt tokens
        for (auto t : tokens) {
            llama_sampler_accept(sampler_state.get(), t);
        }

        // Generation loop
        std::string response;
        int n_generated = 0;
        int64_t cur_pos = n_tokens;

        while (n_generated < max_tokens) {
            llama_token new_token = llama_sampler_sample(sampler_state.get(), ctx, -1);

            if (new_token == llama_vocab_eos(vocab)) {
                break;
            }

            // Convert token to text
            char buf[256];
            int n = llama_token_to_piece(vocab, new_token, buf, (int)sizeof(buf), 0, false);
            if (n > 0) {
                response.append(buf, n);
            }

            llama_sampler_accept(sampler_state.get(), new_token);

            // Decode next token
            llama_batch next_batch = llama_batch_init(1, 0, 1);
            next_batch.n_tokens = 1;
            next_batch.token[0] = new_token;
            next_batch.pos[0] = (llama_pos)cur_pos;
            next_batch.logits[0] = 1;
            next_batch.n_seq_id[0] = 1;
            next_batch.seq_id[0][0] = 0;

            if (llama_decode(ctx, next_batch) != 0) {
                llama_batch_free(next_batch);
                break;
            }
            llama_batch_free(next_batch);

            ++cur_pos;
            ++n_generated;
        }

        return response;
    }
};

std::string create_persona_prompt(const json& input_json) {
    std::string name = input_json["name"];
    std::string position = input_json["position"];
    std::string department = input_json["department"];
    std::string language = input_json["language"];
    
    std::string samples_text;
    if (input_json.contains("samples") && input_json["samples"].is_array()) {
        for (const auto& sample : input_json["samples"]) {
            samples_text += sample.get<std::string>() + " ";
        }
    }
    
    std::string prompt =
    "You are a system that generates professional persona summaries based on writing samples.\n"
    "Use the provided data to infer tone and communication style.\n\n"
    "Name: " + name + "\n"
    "Position: " + position + "\n"
    "Department: " + department + "\n"
    "Language: " + language + "\n"
    "Sample text: \"" + samples_text + "\"\n\n"
    "Analyze the tone (formal / semi-formal / casual) and the communication style (brief, detailed, empathetic, direct, etc.).\n"
    "Then output **exactly one** sentence in the following format:\n\n"
    + name + " (" + position + ", " + department + "). Preferred language: " + language +
    ". [Tone] tone inferred from writing samples. [Brief communication patterns].\n\n"
    "Respond with **only** this single persona sentence. Do not include examples, explanations, or multiple personas.";

    
    return prompt;
}

bool send_to_api(const std::string& text, const std::string& api_url) {
    httplib::Client cli(api_url.c_str());
    cli.set_connection_timeout(10);
    cli.set_read_timeout(30);

    json payload = {{"text", text}};
    std::string body = payload.dump();

    auto res = cli.Post("/ai/profile/persona", body, "application/json");
    if (res && res->status == 200) {
        std::cout << "Sent to API: " << res->body << std::endl;
        return true;
    } else {
        std::cerr << " Status: " << (res ? std::to_string(res->status) : "No response") << std::endl;
        return false;
    }
}

int main() {
    try {
        std::string model_path = "../build/models/google_gemma-3-1b-it-qat-q4_0-gguf_gemma-3-1b-it-q4_0.gguf";
        LlamaInference llama(model_path);
        
        httplib::Server svr;
        
        svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("{\"status\":\"ok\"}", "application/json");
        });
        
        svr.Post("/ai/profile/persona", [&llama](const httplib::Request& req, httplib::Response& res) {
            try {
                // Parse input JSON
                json input_json = json::parse(req.body);
                
                // Validate required fields
                if (!input_json.contains("user_id") || !input_json.contains("name") || 
                    !input_json.contains("position") || !input_json.contains("department") ||
                    !input_json.contains("language") || !input_json.contains("samples")) {
                    res.status = 400;
                    res.set_content("{\"error\":\"Missing required fields\"}", "application/json");
                    return;
                }
                
                std::string user_id = input_json["user_id"];
                
                std::cout << "Processing persona for user_id: " << user_id << std::endl;
                
                // Create prompt from input
                std::string prompt = create_persona_prompt(input_json);
                
                // Generate persona string
                std::string persona_string = llama.generate(prompt, 512);
                
                std::cout << "Raw generated output: [" << persona_string << "]" << std::endl;
                
                // Extract the actual persona line
                // Look for lines that start with the name or contain the pattern "Name (Position"
                std::string name = input_json["name"];
                std::istringstream stream(persona_string);
                std::string line;
                std::string best_line;
                
                while (std::getline(stream, line)) {
                    // Trim the line
                    line.erase(0, line.find_first_not_of(" \n\r\t\""));
                    line.erase(line.find_last_not_of(" \n\r\t\"") + 1);
                    
                    // Skip empty lines, quote markers, or lines with just "Persona:"
                    if (line.empty() || line == "```" || line.find("Persona:") != std::string::npos) {
                        continue;
                    }
                    
                    // Look for a line that starts with the user's name and has good length
                    if (line.find(name) == 0 && line.length() > 50) {
                        best_line = line;
                        break;
                    }
                    
                    // Also accept lines that look like persona descriptions
                    if (line.length() > 50 && line.find("(") != std::string::npos && 
                        line.find(")") != std::string::npos && line.find("Preferred language") != std::string::npos) {
                        best_line = line;
                    }
                }
                
                persona_string = best_line;
                
                // If we still don't have a good persona, create a fallback
                if (persona_string.empty() || persona_string.length() < 20) {
                    std::string position = input_json["position"];
                    std::string department = input_json["department"];
                    std::string language = input_json["language"];
                    
                    persona_string = name + " (" + position + ", " + department + 
                                   "). Preferred language: " + language + 
                                   ". Professional tone inferred from writing samples. Direct communication style.";
                    
                    std::cout << "Used fallback persona" << std::endl;
                }
                
                std::cout << "Final persona: [" << persona_string << "]" << std::endl;
                
                // Send to external API (optional)
                std::string target_api = "http://localhost:8081";
                send_to_api(persona_string, target_api);
                
                // Create output JSON
                json output_json = {
                    {"user_id", user_id},
                    {"persona_string", persona_string}
                };
                
                res.set_content(output_json.dump(), "application/json");
                
            } catch (const json::parse_error& e) {
                res.status = 400;
                res.set_content("{\"error\":\"Invalid JSON: " + std::string(e.what()) + "\"}", 
                              "application/json");
            } catch (const std::exception& e) {
                res.status = 500;
                res.set_content("{\"error\":\"" + std::string(e.what()) + "\"}", 
                              "application/json");
            }
        });
        
        std::cout << "Persona Generation Server starting on port 8080..." << std::endl;
        std::cout << "Endpoint: POST /ai/profile/persona" << std::endl;
        svr.listen("0.0.0.0", 8080);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}