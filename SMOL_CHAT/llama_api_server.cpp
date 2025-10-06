// llama_api_server_persona_debug.cpp
// This version includes extensive debugging to diagnose generation issues

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
#include <mutex>
#include <optional>
#include <iomanip>

using json = nlohmann::json;

class LlamaInference {
private:
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    llama_context_params ctx_params{};
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler_state{nullptr, llama_sampler_free};
    std::mutex inference_mutex;

public:
    LlamaInference(const std::string& model_path, int n_ctx = 2048, int n_threads = 4) {
        std::cout << "[INIT] Starting llama backend..." << std::endl;
        llama_backend_init();

        std::cout << "[INIT] Loading model from: " << model_path << std::endl;
        llama_model_params mparams = llama_model_default_params();
        model = llama_model_load_from_file(model_path.c_str(), mparams);
        if (!model) throw std::runtime_error("Failed to load model from: " + model_path);
        
        std::cout << "[INIT] Model loaded successfully" << std::endl;

        ctx_params = llama_context_default_params();
        ctx_params.n_ctx = n_ctx;
        ctx_params.n_threads = n_threads;
        ctx_params.n_batch = 512;
        
        std::cout << "[INIT] Creating context (n_ctx=" << n_ctx << ", threads=" << n_threads << ")" << std::endl;
        ctx = llama_init_from_model(model, ctx_params);
        if (!ctx) {
            llama_model_free(model);
            throw std::runtime_error("Failed to create context");
        }

        init_sampler();
        std::cout << "[INIT] Initialization complete" << std::endl;
    }

    ~LlamaInference() {
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
        llama_backend_free();
    }

    LlamaInference(const LlamaInference&) = delete;
    LlamaInference& operator=(const LlamaInference&) = delete;

    std::string generate(const std::string& prompt, int max_tokens = 512) {
        std::lock_guard<std::mutex> lock(inference_mutex);
        
        std::cout << "\n[GENERATE] Starting generation..." << std::endl;
        std::cout << "[GENERATE] Prompt length: " << prompt.length() << " chars" << std::endl;
        std::cout << "[GENERATE] Prompt preview: " << prompt.substr(0, std::min(size_t(200), prompt.length())) << "..." << std::endl;
        
        if (!model || !ctx) throw std::runtime_error("Model or context not initialized");

        // Clear context and reset sampler
        std::cout << "[GENERATE] Clearing context..." << std::endl;
        llama_memory_clear(llama_get_memory(ctx), false);
        llama_sampler_reset(sampler_state.get());
        
        const llama_model* model_info = llama_get_model(ctx);
        const llama_vocab* vocab = llama_model_get_vocab(model_info);

        // Tokenize prompt
        std::cout << "[GENERATE] Tokenizing prompt..." << std::endl;
        std::vector<llama_token> tokens = tokenize_prompt(vocab, prompt);
        std::cout << "[GENERATE] Tokenized to " << tokens.size() << " tokens" << std::endl;

        // Check if tokens fit in context
        if (tokens.size() >= ctx_params.n_ctx) {
            std::cerr << "[ERROR] Prompt too long! " << tokens.size() << " tokens exceeds context size " << ctx_params.n_ctx << std::endl;
            throw std::runtime_error("Prompt exceeds context size");
        }

        // Decode prompt
        std::cout << "[GENERATE] Decoding prompt..." << std::endl;
        decode_prompt(tokens);
        std::cout << "[GENERATE] Prompt decoded successfully" << std::endl;

        // Make sampler aware of prompt tokens
        for (auto t : tokens) {
            llama_sampler_accept(sampler_state.get(), t);
        }

        // Generation loop
        std::cout << "[GENERATE] Starting token generation (max_tokens=" << max_tokens << ")..." << std::endl;
        std::string result = generate_tokens(vocab, tokens.size(), max_tokens);
        std::cout << "[GENERATE] Generation complete. Generated " << result.length() << " characters" << std::endl;
        
        return result;
    }

private:
    void init_sampler() {
        std::cout << "[INIT] Initializing sampler chain..." << std::endl;
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
        std::cout << "[INIT] Sampler chain configured (top_k=40, top_p=0.9, temp=0.7)" << std::endl;
    }

    std::vector<llama_token> tokenize_prompt(const llama_vocab* vocab, const std::string& prompt) {
        std::vector<llama_token> tokens;
        tokens.resize(prompt.size() * 4 + 16);

        int n_tokens = llama_tokenize(vocab,
                                     prompt.c_str(), (int)prompt.size(),
                                     tokens.data(), (int)tokens.size(),
                                     true,  // add_special
                                     false); // parse_special
        
        if (n_tokens < 0) {
            std::cerr << "[ERROR] Tokenization failed with code: " << n_tokens << std::endl;
            throw std::runtime_error("Tokenization failed");
        }
        
        tokens.resize(n_tokens);
        
        // Debug: print first few tokens
        std::cout << "[TOKENIZE] First few tokens: ";
        for (size_t i = 0; i < std::min(size_t(10), tokens.size()); ++i) {
            std::cout << tokens[i] << " ";
        }
        std::cout << std::endl;
        
        return tokens;
    }

    void decode_prompt(const std::vector<llama_token>& tokens) {
        llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
        batch.n_tokens = tokens.size();
        
        for (size_t i = 0; i < tokens.size(); ++i) {
            batch.token[i]   = tokens[i];
            batch.pos[i]     = i;
            batch.logits[i]  = (i == tokens.size() - 1);  // Only last token needs logits
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
        }
        
        int decode_result = llama_decode(ctx, batch);
        llama_batch_free(batch);
        
        if (decode_result != 0) {
            std::cerr << "[ERROR] Decode failed with code: " << decode_result << std::endl;
            throw std::runtime_error("Failed to decode prompt");
        }
    }

    std::string generate_tokens(const llama_vocab* vocab, size_t prompt_length, int max_tokens) {
        std::string response;
        int n_generated = 0;
        int64_t cur_pos = prompt_length;
        int eos_count = 0;

        while (n_generated < max_tokens) {
            llama_token new_token = llama_sampler_sample(sampler_state.get(), ctx, -1);
            
            // Debug logging every 10 tokens
            if (n_generated % 10 == 0 || n_generated < 5) {
                std::cout << "[GEN] Token " << n_generated << ": " << new_token << std::endl;
            }

            // Check for EOS
            if (new_token == llama_vocab_eos(vocab)) {
                eos_count++;
                std::cout << "[GEN] EOS token encountered at position " << n_generated << std::endl;
                if (eos_count >= 1) {  // Stop on first EOS
                    break;
                }
            }

            // Check for invalid tokens
            if (new_token < 0) {
                std::cerr << "[ERROR] Invalid token sampled: " << new_token << std::endl;
                break;
            }

            // Convert token to text
            char buf[256];
            int n = llama_token_to_piece(vocab, new_token, buf, (int)sizeof(buf), 0, false);
            if (n > 0) {
                std::string piece(buf, n);
                response.append(piece);
                
                // Debug: print first few pieces
                if (n_generated < 20) {
                    std::cout << "[GEN] Piece " << n_generated << ": \"" << piece << "\"" << std::endl;
                }
            } else {
                std::cerr << "[WARN] token_to_piece returned " << n << " for token " << new_token << std::endl;
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

            int decode_result = llama_decode(ctx, next_batch);
            llama_batch_free(next_batch);
            
            if (decode_result != 0) {
                std::cerr << "[ERROR] Decode failed at token " << n_generated << " with code " << decode_result << std::endl;
                break;
            }

            ++cur_pos;
            ++n_generated;
        }

        std::cout << "[GEN] Generation loop completed. Tokens generated: " << n_generated << std::endl;
        std::cout << "[GEN] Response length: " << response.length() << " characters" << std::endl;
        
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
    
    // Simplified prompt for better results with smaller models
    std::string prompt =
    "Generate a one-sentence professional persona summary.\n\n"
    "Input:\n"
    "Name: " + name + "\n"
    "Position: " + position + "\n"
    "Department: " + department + "\n"
    "Language: " + language + "\n"
    "Writing samples: " + samples_text + "\n\n"
    "Output format:it should include these fild specifically\n"
    + name + " (" + position + ", " + department + "). Preferred language: " + language + ". [tone] tone. [style] communication style.\n\n"
    "Persona:";
    
    return prompt;
}

std::string extract_persona_line(const std::string& raw_output, const std::string& name) {
    if (raw_output.empty()) {
        std::cout << "[EXTRACT] Empty raw output" << std::endl;
        return "";
    }
    
    std::cout << "[EXTRACT] Processing output of length " << raw_output.length() << std::endl;
    
    std::istringstream stream(raw_output);
    std::string line;
    std::string best_line;
    int line_count = 0;
    
    while (std::getline(stream, line)) {
        line_count++;
        
        // Trim the line
        line.erase(0, line.find_first_not_of(" \n\r\t\""));
        line.erase(line.find_last_not_of(" \n\r\t\"") + 1);
        
        std::cout << "[EXTRACT] Line " << line_count << " (len=" << line.length() << "): \"" << line.substr(0, 80) << "...\"" << std::endl;
        
        // Skip empty lines, code blocks, or metadata
        if (line.empty() || line == "```" || line.find("Persona:") != std::string::npos) {
            continue;
        }
        
        // Look for a line that starts with the user's name
        if (line.find(name) == 0 && line.length() > 50) {
            best_line = line;
            std::cout << "[EXTRACT] Found matching line starting with name" << std::endl;
            break;
        }
        
        // Accept lines that look like persona descriptions
        if (line.length() > 50 && line.find("(") != std::string::npos && 
            line.find(")") != std::string::npos) {
            best_line = line;
            std::cout << "[EXTRACT] Found potential persona line" << std::endl;
        }
    }
    
    std::cout << "[EXTRACT] Processed " << line_count << " lines" << std::endl;
    
    return best_line;
}

std::string create_fallback_persona(const json& input_json) {
    std::string name = input_json["name"];
    std::string position = input_json["position"];
    std::string department = input_json["department"];
    std::string language = input_json["language"];
    
    return name + " (" + position + ", " + department + 
           "). Preferred language: " + language + 
           ". Professional tone inferred from writing samples. Direct communication style.";
}

std::optional<std::string> send_to_api(const std::string& text, const std::string& api_url) {
    try {
        std::cout << "[API] Attempting to send to: " << api_url << std::endl;
        
        httplib::Client cli(api_url.c_str());
        cli.set_connection_timeout(5);  // Reduced timeout
        cli.set_read_timeout(10);

        json payload = {{"text", text}};
        std::string body = payload.dump();

        auto res = cli.Post("/ai/profile/persona", body, "application/json");
        if (res && res->status == 200) {
            std::cout << "[API] Success: " << res->body << std::endl;
            return res->body;
        } else {
            std::cerr << "[API] Failed. Status: " << (res ? std::to_string(res->status) : "No response") << std::endl;
            return std::nullopt;
        }
    } catch (const std::exception& e) {
        std::cerr << "[API] Exception: " << e.what() << std::endl;
        return std::nullopt;
    }
}

int main(int argc, char* argv[]) {
    try {
        std::string model_path = "../build/models/google_gemma-3-1b-it-qat-q4_0-gguf_gemma-3-1b-it-q4_0.gguf";
        if (argc > 1) {
            model_path = argv[1];
        }
        
        std::cout << "========================================" << std::endl;
        std::cout << "Persona Generation Server (Debug Mode)" << std::endl;
        std::cout << "========================================" << std::endl;
        
        LlamaInference llama(model_path);
        
        httplib::Server svr;
        
        svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("{\"status\":\"ok\"}", "application/json");
        });
        
        svr.Post("/ai/profile/persona", [&llama](const httplib::Request& req, httplib::Response& res) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "NEW REQUEST RECEIVED" << std::endl;
            std::cout << "========================================" << std::endl;
            
            try {
                json input_json = json::parse(req.body);
                
                std::cout << "[REQUEST] Body: " << input_json.dump(2) << std::endl;
                
                // Validate required fields
                std::vector<std::string> required_fields = {"user_id", "name", "position", "department", "language", "samples"};
                for (const auto& field : required_fields) {
                    if (!input_json.contains(field)) {
                        res.status = 400;
                        json error_response = {{"error", "Missing required field: " + field}};
                        res.set_content(error_response.dump(), "application/json");
                        return;
                    }
                }
                
                std::string user_id = input_json["user_id"];
                std::string name = input_json["name"];
                
                std::cout << "[REQUEST] Processing for user: " << name << " (ID: " << user_id << ")" << std::endl;
                
                std::string prompt = create_persona_prompt(input_json);
                std::cout << "[REQUEST] Prompt created (" << prompt.length() << " chars)" << std::endl;
                
                std::string raw_output = llama.generate(prompt, 256);  // Reduced max_tokens
                
                std::cout << "\n[OUTPUT] Raw generated output:" << std::endl;
                std::cout << "----------------------------------------" << std::endl;
                std::cout << raw_output << std::endl;
                std::cout << "----------------------------------------" << std::endl;
                
                std::string persona_string = extract_persona_line(raw_output, name);
                
                if (persona_string.empty() || persona_string.length() < 20) {
                    persona_string = create_fallback_persona(input_json);
                    std::cout << "[RESULT] Using fallback persona" << std::endl;
                } else {
                    std::cout << "[RESULT] Successfully extracted persona" << std::endl;
                }
                
                std::cout << "[RESULT] Final persona: " << persona_string << std::endl;
                
                // Optional external API call
                std::string target_api = "http://localhost:8081";
                send_to_api(persona_string, target_api);
                
                json output_json = {
                    {"user_id", user_id},
                    {"persona_string", persona_string}
                };
                
                res.set_content(output_json.dump(), "application/json");
                std::cout << "[REQUEST] Response sent successfully\n" << std::endl;
                
            } catch (const json::parse_error& e) {
                res.status = 400;
                json error_response = {{"error", "Invalid JSON"}, {"details", e.what()}};
                res.set_content(error_response.dump(), "application/json");
            } catch (const std::exception& e) {
                res.status = 500;
                json error_response = {{"error", "Internal server error"}, {"details", e.what()}};
                res.set_content(error_response.dump(), "application/json");
            }
        });
        
        std::cout << "\n[SERVER] Starting on port 8080..." << std::endl;
        std::cout << "[SERVER] Endpoints:" << std::endl;
        std::cout << "  - POST /ai/profile/persona" << std::endl;
        std::cout << "  - GET  /health" << std::endl;
        std::cout << "========================================\n" << std::endl;
        
        svr.listen("0.0.0.0", 8080);
        
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}