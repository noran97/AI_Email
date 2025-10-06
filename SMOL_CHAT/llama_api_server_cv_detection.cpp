#include "httplib.h"
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <fstream>
#include <algorithm> 

// POSIX/Linux Headers for temp files and directory manipulation
#include <sys/stat.h>
#include <sys/types.h>
#include <array>
#include <cstdio>
#include <unistd.h>
#include <fcntl.h>

// For PDF to image conversion
#include <poppler/cpp/poppler-document.h>
#include <poppler/cpp/poppler-page.h>
#include <poppler/cpp/poppler-page-renderer.h>
#include <poppler/cpp/poppler-image.h>

using json = nlohmann::json;

// Execute command and capture output
std::string exec_command(const std::string& cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    
    return result;
}

// Cleanup helper function
void cleanup_temp_images(const std::vector<std::string>& image_paths) {
    for (const auto& path : image_paths) {
        if (!path.empty()) {
            remove(path.c_str());
        }
    }
}

std::string get_cli_version(const std::string& llama_cli_path) {
    std::string version_cmd = llama_cli_path + " --version 2>&1";
    std::string version_output;
    try {
        version_output = exec_command(version_cmd);
        size_t first = version_output.find_first_not_of(" \t\n\r");
        size_t last = version_output.find_last_not_of(" \t\n\r");
        if (std::string::npos == first || std::string::npos == last) {
            return "Version check failed or empty output.";
        }
        return version_output.substr(first, (last - first + 1));
    } catch (const std::exception& e) {
        return "Version check failed: " + std::string(e.what());
    }
}

bool is_pdf_file(const std::string& filename) {
    if (filename.length() < 4) return false;
    std::string ext = filename.substr(filename.length() - 4);
    for (auto& c : ext) c = std::tolower(c);
    return ext == ".pdf";
}

std::string pdf_to_image(const std::string& pdf_path, const std::string& output_dir) {
    struct stat pdf_stat;
    if (stat(pdf_path.c_str(), &pdf_stat) != 0) {
         throw std::runtime_error("PDF file not found at: " + pdf_path);
    }
    
    std::unique_ptr<poppler::document> doc(poppler::document::load_from_file(pdf_path));
    if (!doc || doc->is_locked()) {
        throw std::runtime_error("Cannot open or read PDF: " + pdf_path);
    }
    
    std::unique_ptr<poppler::page> page(doc->create_page(0));
    if (!page) {
        throw std::runtime_error("Cannot read first page of PDF");
    }
    
    poppler::page_renderer renderer;
    renderer.set_render_hint(poppler::page_renderer::text_antialiasing);
    renderer.set_render_hint(poppler::page_renderer::antialiasing);
    
    poppler::image img = renderer.render_page(page.get(), 150, 150);
    
    if (!img.is_valid()) {
        throw std::runtime_error("Failed to render PDF page to image");
    }
    
    std::string base_name = pdf_path.substr(pdf_path.find_last_of("/\\") + 1);
    base_name = base_name.substr(0, base_name.find_last_of('.'));
    std::string output_path = output_dir + "/" + base_name + "_page1.png";
    
    if (!img.save(output_path, "png")) {
        throw std::runtime_error("Failed to save image: " + output_path);
    }
    
    std::cout << "Converted PDF to image: " << output_path << std::endl;
    return output_path;
}

std::string create_cv_detection_prompt() {
    std::string prompt = 
        "You are an AI assistant that extracts information from CV/resume images.\\n\\n"
        "Please analyze the CV image and extract the following information:\\n"
        "1. Name (full name of the candidate)\\n"
        "2. Position (job title or desired position)\\n"
        "3. Skills (list up to 10 key technical skills)\\n"
        "4. Experience (total years of professional experience)\\n"
        "5. Education (highest degree)\\n\\n"
        "Return ONLY valid JSON in this exact format with no additional text:\\n"
        "{\\n"
        "  \\\"name\\\": \\\"Full Name\\\",\\n"
        "  \\\"position\\\": \\\"Job Title\\\",\\n"
        "  \\\"skills\\\": [\\\"skill1\\\", \\\"skill2\\\", \\\"skill3\\\"],\\n"
        "  \\\"experience\\\": \\\"X years\\\",\\n"
        "  \\\"education\\\": \\\"Degree Name\\\"\\n"
        "}\\n\\n"
        "Output:";
    return prompt;
}

std::string create_draft_reply_prompt(const std::string& persona_string, 
                                      const std::string& subject,
                                      const std::string& body,
                                      const std::string& instruction,
                                      bool has_attachments) {
    std::string prompt = 
        "You are an AI assistant that drafts email replies based on user persona and instructions.\\n\\n"
        "Persona: " + persona_string + "\\n\\n"
        "Original Email Subject: " + subject + "\\n"
        "Original Email Body: " + body + "\\n\\n";
    
    if (has_attachments) {
        prompt += "Note: The email contains attachments (images shown above represent PDF content).\\n\\n";
    }
    
    // Only add instruction if it's not empty
    if (!instruction.empty()) {
        prompt += "Instruction: " + instruction + "\\n\\n";
    }
    
    prompt += "Draft a reply email that:\\n"
        "1. Matches the persona's tone and language preference\\n"
        "2. ";
    
    // Adjust prompt based on whether instruction is provided
    if (!instruction.empty()) {
        prompt += "Follows the given instruction\\n";
    } else {
        prompt += "Provides an appropriate response to the original email\\n";
    }
    
    prompt += "3. References attachment content if relevant\\n"
        "4. Is professional and appropriate\\n\\n"
        "Return ONLY valid JSON in this exact format with no additional text:\\n"
        "{\\n"
        "  \\\"subject\\\": \\\"Re: [original subject]\\\",\\n"
        "  \\\"draft_reply\\\": \\\"Your drafted email reply here\\\"\\n"
        "}\\n\\n"
        "Output:";
    
    return prompt;
}

json parse_cv_metadata(const std::string& model_output) {
    size_t start_marker = model_output.find("```json");
    if (start_marker == std::string::npos) {
        start_marker = model_output.find('{');
    } else {
        start_marker += 7;
        while (start_marker < model_output.length() && 
               (model_output[start_marker] == '\n' || model_output[start_marker] == '\r' || 
                model_output[start_marker] == ' ')) {
            start_marker++;
        }
    }

    size_t end_marker = model_output.rfind('}');
    
    if (start_marker != std::string::npos && end_marker != std::string::npos && 
        end_marker > start_marker) {
        std::string json_str = model_output.substr(start_marker, end_marker - start_marker + 1);

        while (!json_str.empty() && 
               (json_str.back() == '`' || json_str.back() == '\n' || 
                json_str.back() == '\r' || json_str.back() == ' ')) {
             json_str.pop_back();
        }
        
        size_t npos;
        std::string non_breaking_space_utf8 = "\xC2\xA0"; 
        
        while ((npos = json_str.find(non_breaking_space_utf8)) != std::string::npos) {
            json_str.replace(npos, non_breaking_space_utf8.length(), " "); 
        }

        try {
            return json::parse(json_str);
        } catch (const json::parse_error& e) {
            std::cerr << "JSON parse error (Cleaned string failed): " << e.what() << std::endl;
            std::cerr << "Attempted to parse: " << json_str << std::endl;
        }
    } else {
        std::cerr << "JSON delimiters not found or invalid range in model output." << std::endl;
    }
    
    return json{
        {"name", "Unknown"}, {"position", "Unknown"}, {"skills", json::array()},
        {"experience", "Unknown"}, {"education", "Unknown"}
    };
}

//  Parse draft reply response
json parse_draft_reply(const std::string& model_output) {
    size_t start_marker = model_output.find("```json");
    if (start_marker == std::string::npos) {
        start_marker = model_output.find('{');
    } else {
        start_marker += 7;
        while (start_marker < model_output.length() && 
               (model_output[start_marker] == '\n' || model_output[start_marker] == '\r' || 
                model_output[start_marker] == ' ')) {
            start_marker++;
        }
    }

    size_t end_marker = model_output.rfind('}');
    
    if (start_marker != std::string::npos && end_marker != std::string::npos && 
        end_marker > start_marker) {
        std::string json_str = model_output.substr(start_marker, end_marker - start_marker + 1);

        while (!json_str.empty() && 
               (json_str.back() == '`' || json_str.back() == '\n' || 
                json_str.back() == '\r' || json_str.back() == ' ')) {
             json_str.pop_back();
        }
        
        size_t npos;
        std::string non_breaking_space_utf8 = "\xC2\xA0"; 
        
        while ((npos = json_str.find(non_breaking_space_utf8)) != std::string::npos) {
            json_str.replace(npos, non_breaking_space_utf8.length(), " "); 
        }

        try {
            return json::parse(json_str);
        } catch (const json::parse_error& e) {
            std::cerr << "JSON parse error: " << e.what() << std::endl;
            std::cerr << "Attempted to parse: " << json_str << std::endl;
        }
    }
    
    return json{
        {"subject", "Re: [Subject]"},
        {"draft_reply", "Unable to generate reply. Please try again."}
    };
}
std::string create_classification_prompt(const std::string& subject,
                                         const std::string& body,
                                         bool has_attachments) {
    std::string prompt = 
        "You are an AI assistant that classifies emails based on urgency and priority.\\n\\n"
        "Email Subject: " + subject + "\\n"
        "Email Body: " + body + "\\n\\n";
    
    if (has_attachments) {
        prompt += "Note: The email contains attachments (images shown above represent PDF content).\\n\\n";
    }
    
    prompt += "Classify this email into ONE of the following categories:\\n"
        "1. \\\"Urgent & Action Required\\\" - Requires immediate attention and action\\n"
        "2. \\\"Normal Follow-up\\\" - Regular business communication requiring response\\n"
        "3. \\\"FYI / Low Priority\\\" - Informational only, no immediate action needed\\n"
        "4. \\\"Spam\\\" - Unsolicited, irrelevant, or suspicious content\\n\\n"
        "Consider:\\n"
        "- Time-sensitive keywords (deadline, urgent, ASAP, today, tomorrow)\\n"
        "- Action verbs (submit, complete, respond, approve)\\n"
        "- Sender context and attachment relevance\\n\\n"
        "Return ONLY valid JSON in this exact format with no additional text:\\n"
        "{\\n"
        "  \\\"category\\\": \\\"One of the four categories above\\\",\\n"
        "  \\\"confidence\\\": 0.85\\n"
        "}\\n\\n"
        "Output:";
    
    return prompt;
}
json parse_classification(const std::string& model_output) {
    size_t start_marker = model_output.find("```json");
    if (start_marker == std::string::npos) {
        start_marker = model_output.find('{');
    } else {
        start_marker += 7;
        while (start_marker < model_output.length() && 
               (model_output[start_marker] == '\n' || model_output[start_marker] == '\r' || 
                model_output[start_marker] == ' ')) {
            start_marker++;
        }
    }

    size_t end_marker = model_output.rfind('}');
    
    if (start_marker != std::string::npos && end_marker != std::string::npos && 
        end_marker > start_marker) {
        std::string json_str = model_output.substr(start_marker, end_marker - start_marker + 1);

        while (!json_str.empty() && 
               (json_str.back() == '`' || json_str.back() == '\n' || 
                json_str.back() == '\r' || json_str.back() == ' ')) {
             json_str.pop_back();
        }
        
        size_t npos;
        std::string non_breaking_space_utf8 = "\xC2\xA0"; 
        
        while ((npos = json_str.find(non_breaking_space_utf8)) != std::string::npos) {
            json_str.replace(npos, non_breaking_space_utf8.length(), " "); 
        }

        try {
            json parsed = json::parse(json_str);
            
            // Validate category
            std::string category = parsed.value("category", "FYI / Low Priority");
            std::vector<std::string> valid_categories = {
                "Urgent & Action Required",
                "Normal Follow-up",
                "FYI / Low Priority",
                "Spam"
            };
            
            bool valid = false;
            for (const auto& valid_cat : valid_categories) {
                if (category == valid_cat) {
                    valid = true;
                    break;
                }
            }
            
            if (!valid) {
                category = "FYI / Low Priority";
            }
            
            double confidence = parsed.value("confidence", 0.5);
            if (confidence < 0.0) confidence = 0.0;
            if (confidence > 1.0) confidence = 1.0;
            
            return json{
                {"category", category},
           {"confidence", confidence}
            };
            
        } catch (const json::parse_error& e) {
            std::cerr << "JSON parse error: " << e.what() << std::endl;
            std::cerr << "Attempted to parse: " << json_str << std::endl;
        }
    }
    
    return json{
        {"category", "FYI / Low Priority"},
        {"confidence", 0.5}
    };
}
std::string process_cv_with_vision(const std::vector<std::string>& image_paths, 
                                   const std::string& llama_cli_path, 
                                   const std::string& main_model_path, 
                                   const std::string& mmproj_path) {
    
    std::string prompt = create_cv_detection_prompt();
    
    std::string image_args;
    for (const auto& path : image_paths) {
        image_args += " --image " + path;
        std::cout << "  Passing image: " << path << std::endl;
    }
    
    std::string cmd = llama_cli_path + " " +
                      "-m " + main_model_path + " " + 
                      "--mmproj " + mmproj_path + " " +
                      image_args + " " +
                      "-p \"" + prompt + "\" " + 
                      "--n-gpu-layers 0 " + 
                      "--temp 0.3 " +
                      "-n 800 " +
                      "2>&1";   
        
    std::cout << "Executing vision model..." << std::endl;
    std::cout << "Command: " << cmd << std::endl;
    
    try {
        std::string output = exec_command(cmd);
        std::cout << "Vision model raw output: " << output << std::endl;
        return output;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to execute vision model: " + std::string(e.what()));
    }
}

// NEW: Process email with vision model for draft reply
std::string process_draft_reply_with_vision(const std::vector<std::string>& image_paths,
                                            const std::string& persona_string,
                                            const std::string& subject,
                                            const std::string& body,
                                            const std::string& instruction,
                                            const std::string& llama_cli_path, 
                                            const std::string& main_model_path, 
                                            const std::string& mmproj_path) {
    
    std::string prompt = create_draft_reply_prompt(persona_string, subject, body, 
                                                   instruction, !image_paths.empty());
    
    std::string image_args;
    for (const auto& path : image_paths) {
        image_args += " --image " + path;
        std::cout << "  Passing image: " << path << std::endl;
    }
    
    std::string cmd = llama_cli_path + " " +
                      "-m " + main_model_path + " " + 
                      "--mmproj " + mmproj_path + " " +
                      image_args + " " +
                      "-p \"" + prompt + "\" " + 
                      "--n-gpu-layers 0 " + 
                      "--temp 0.7 " +
                      "-n 1000 " +
                      "2>&1";   
        
    std::cout << "Executing vision model for draft reply..." << std::endl;
    std::cout << "Command: " << cmd << std::endl;
    
    try {
        std::string output = exec_command(cmd);
        std::cout << "Vision model raw output: " << output << std::endl;
        return output;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to execute vision model: " + std::string(e.what()));
    }
}
std::string process_classification_with_vision(const std::vector<std::string>& image_paths,
                                               const std::string& subject,
                                               const std::string& body,
                                               const std::string& llama_cli_path, 
                                               const std::string& main_model_path, 
                                               const std::string& mmproj_path) {
    
    std::string prompt = create_classification_prompt(subject, body, !image_paths.empty());
    
    std::string image_args;
    for (const auto& path : image_paths) {
        image_args += " --image " + path;
        std::cout << "  Passing image: " << path << std::endl;
    }
    
    std::string cmd = llama_cli_path + " " +
                      "-m " + main_model_path + " " + 
                      "--mmproj " + mmproj_path + " " +
                      image_args + " " +
                      "-p \"" + prompt + "\" " + 
                      "--n-gpu-layers 0 " + 
                      "--temp 0.3 " +
                      "-n 500 " +
                      "2>&1";   
        
    std::cout << "Executing vision model for classification..." << std::endl;
    std::cout << "Command: " << cmd << std::endl;
    
    try {
        std::string output = exec_command(cmd);
        std::cout << "Vision model raw output: " << output << std::endl;
        return output;
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to execute vision model: " + std::string(e.what()));
    }
}
int main(int argc, char** argv) {
    try {
        // Configuration
        std::string main_model_path = "/home/nor/.cache/llama.cpp/google_gemma-3-4b-it-qat-q4_0-gguf_gemma-3-4b-it-q4_0.gguf";
        std::string mmproj_path = "/home/nor/.cache/llama.cpp/google_gemma-3-4b-it-qat-q4_0-gguf_mmproj-model-f16-4B.gguf"; 
        std::string llama_cli_path = "../externals/llama.cpp/build/bin/llama-mtmd-cli";
        
        // Parse command line arguments
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--main-model-path" && i + 1 < argc) {
                main_model_path = argv[++i]; 
            } else if (arg == "--mmproj-path" && i + 1 < argc) {
                mmproj_path = argv[++i];
            } else if (arg == "--cli-path" && i + 1 < argc) {
                llama_cli_path = argv[++i];
            }
        }
        
        // Check local model and CLI files
        auto check_file = [](const std::string& path, const std::string& name) {
            struct stat stat_buffer;
            if (stat(path.c_str(), &stat_buffer) != 0) {
                std::cerr << "ERROR: Local " << name << " file not found at: " << path << std::endl;
                std::cerr << "Please ensure the file exists." << std::endl;
                return false;
            }
            return true;
        };

        if (!check_file(main_model_path, "main model") || 
            !check_file(mmproj_path, "multimodal projection")) {
            return 1;
        }
        
        struct stat cli_stat;
        if (stat(llama_cli_path.c_str(), &cli_stat) != 0) {
            std::cerr << "ERROR: llama-mtmd-cli not found at: " << llama_cli_path << std::endl;
            std::cerr << "Please build it first or specify correct path with --cli-path" << std::endl;
            return 1;
        }
        
        std::string cli_version = get_cli_version(llama_cli_path);
        
        std::cout << "Configuration:" << std::endl;
        std::cout << "  CLI Version: " << cli_version << std::endl;
        std::cout << "  Main Model Path: " << main_model_path << std::endl;
        std::cout << "  MMProj Path: " << mmproj_path << std::endl;
        std::cout << "  CLI Path: " << llama_cli_path << std::endl;
        
        httplib::Server svr;
        svr.set_payload_max_length(10 * 1024 * 1024);
        
        svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            res.set_content("{\"status\":\"ok\"}", "application/json");
        });
        
        // CV Detection Endpoint
        svr.Post("/ai/inbox/detect-cv", [main_model_path, mmproj_path, &llama_cli_path](
            const httplib::Request& req, httplib::Response& res) {
            std::vector<std::string> image_paths; 
            bool cv_detected = false;
            
            try {
                json input_json = json::parse(req.body);
                
                if (!input_json.contains("attachments")) {
                    res.status = 400;
                    res.set_content("{\"error\":\"Missing required fields: attachments\"}", 
                                    "application/json");
                    return;
                }
                
                std::string email_id = input_json["email_id"];
                json attachments = input_json["attachments"];
                json metadata;

                for (const auto& attachment : attachments) {
                    std::string filename = attachment.get<std::string>();
                    std::cout << "Checking attachment: " << filename << std::endl;

                    if (is_pdf_file(filename)) {
                        try {
                            std::string pdf_path = "../uploads/" + filename;
                            std::string temp_dir = "../uploads/temp";
                            
                            struct stat st = {0};
                            if (stat(temp_dir.c_str(), &st) == -1) {
                                if (mkdir(temp_dir.c_str(), 0755) != 0) {
                                    throw std::runtime_error("Failed to create temp directory");
                                }
                            }
                            
                            std::string current_image_path = pdf_to_image(pdf_path, temp_dir); 
                            image_paths.push_back(current_image_path);
                            
                        } catch (const std::exception& e) {
                            std::cerr << "Error converting PDF " << filename << ": " 
                                     << e.what() << std::endl;
                            continue;
                        }
                    }
                }
                
                if (!image_paths.empty()) {
                    cv_detected = true;
                    std::string model_output = process_cv_with_vision(image_paths, llama_cli_path, 
                                                                      main_model_path, mmproj_path);
                    metadata = parse_cv_metadata(model_output);
                } else {
                    metadata = json::object();
                }

                cleanup_temp_images(image_paths);
                
                json output_json = {
                    {"email_id", email_id},
                    {"cv_detected", cv_detected}
                };
                output_json["metadata"] = metadata;
                
                res.set_content(output_json.dump(2), "application/json");
                
            } catch (const std::exception& e) {
                cleanup_temp_images(image_paths);
                res.status = 500;
                res.set_content("{\"error\":\"" + std::string(e.what()) + "\"}", 
                               "application/json");
            }
        });
    svr.Post("/ai/inbox/draft-reply", [main_model_path, mmproj_path, &llama_cli_path](
    const httplib::Request& req, httplib::Response& res) {
    std::vector<std::string> image_paths;
    
    try {
        json input_json = json::parse(req.body);
        
        // Validate required fields (instruction is now optional)
        if (!input_json.contains("email_id") || !input_json.contains("subject") || 
            !input_json.contains("body") || !input_json.contains("persona_string")) {
            res.status = 400;
            res.set_content("{\"error\":\"Missing required fields: email_id, subject, body, persona_string\"}", 
                           "application/json");
            return;
        }
        
        std::string email_id = input_json["email_id"];
        std::string subject = input_json["subject"];
        std::string body = input_json["body"];
        std::string persona_string = input_json["persona_string"];
        
        // Instruction is now optional - default to empty string if not provided
        std::string instruction = input_json.value("instruction", "");
        
        // Process attachments if present
        if (input_json.contains("attachments") && input_json["attachments"].is_array()) {
            json attachments = input_json["attachments"];
            
            for (const auto& attachment : attachments) {
                if (!attachment.contains("filename")) continue;
                
                std::string filename = attachment["filename"].get<std::string>();
                std::cout << "Processing attachment: " << filename << std::endl;

                if (is_pdf_file(filename)) {
                    try {
                        std::string pdf_path = "../uploads/" + filename;
                        std::string temp_dir = "../uploads/temp";
                        
                        struct stat st = {0};
                        if (stat(temp_dir.c_str(), &st) == -1) {
                            if (mkdir(temp_dir.c_str(), 0755) != 0) {
                                throw std::runtime_error("Failed to create temp directory");
                            }
                        }
                        
                        std::string current_image_path = pdf_to_image(pdf_path, temp_dir);
                        image_paths.push_back(current_image_path);
                        
                    } catch (const std::exception& e) {
                        std::cerr << "Error converting PDF " << filename << ": " 
                                 << e.what() << std::endl;
                        continue;
                    }
                }
            }
        }
        
        // Generate draft reply
        std::string model_output = process_draft_reply_with_vision(
            image_paths, persona_string, subject, body, instruction,
            llama_cli_path, main_model_path, mmproj_path
        );
        
        json reply_data = parse_draft_reply(model_output);
        
        cleanup_temp_images(image_paths);
        
        json output_json = {
            {"email_id", email_id},
            {"subject", reply_data["subject"]},
            {"draft_reply", reply_data["draft_reply"]}
        };
        
        res.set_content(output_json.dump(2), "application/json");
        
    } catch (const std::exception& e) {
        cleanup_temp_images(image_paths);
        res.status = 500;
        res.set_content("{\"error\":\"" + std::string(e.what()) + "\"}", 
                       "application/json");
    }
});
        svr.Post("/ai/inbox/classify", [main_model_path, mmproj_path, &llama_cli_path](
            const httplib::Request& req, httplib::Response& res) {
            std::vector<std::string> image_paths;
            
            try {
                json input_json = json::parse(req.body);
                
                // Validate required fields
                if (!input_json.contains("email_id") || !input_json.contains("subject") || 
                    !input_json.contains("body")) {
                    res.status = 400;
                    res.set_content("{\"error\":\"Missing required fields: email_id, subject, body\"}", 
                                   "application/json");
                    return;
                }
                
                std::string email_id = input_json["email_id"];
                std::string subject = input_json["subject"];
                std::string body = input_json["body"];
                
                // Process attachments if present (optional)
                if (input_json.contains("attachments") && input_json["attachments"].is_array()) {
                    json attachments = input_json["attachments"];
                    
                    for (const auto& attachment : attachments) {
                        if (!attachment.contains("filename")) continue;
                        
                        std::string filename = attachment["filename"].get<std::string>();
                        std::cout << "Processing attachment for classification: " << filename << std::endl;

                        if (is_pdf_file(filename)) {
                            try {
                                std::string pdf_path = "../uploads/" + filename;
                                std::string temp_dir = "../uploads/temp";
                                
                                struct stat st = {0};
                                if (stat(temp_dir.c_str(), &st) == -1) {
                                    if (mkdir(temp_dir.c_str(), 0755) != 0) {
                                        throw std::runtime_error("Failed to create temp directory");
                                    }
                                }
                                
                                std::string current_image_path = pdf_to_image(pdf_path, temp_dir);
                                image_paths.push_back(current_image_path);
                                
                            } catch (const std::exception& e) {
                                std::cerr << "Error converting PDF " << filename << ": " 
                                         << e.what() << std::endl;
                                continue;
                            }
                        }
                    }
                }
                
                // Classify email
                std::string model_output = process_classification_with_vision(
                    image_paths, subject, body,
                    llama_cli_path, main_model_path, mmproj_path
                );
                
                json classification_data = parse_classification(model_output);
                
                cleanup_temp_images(image_paths);
                
                json output_json = {
                    {"email_id", email_id},
                    {"category", classification_data["category"]},
                    {"confidence", classification_data["confidence"]}
                };
                
                res.set_content(output_json.dump(2), "application/json");
                
            } catch (const std::exception& e) {
                cleanup_temp_images(image_paths);
                res.status = 500;
                res.set_content("{\"error\":\"" + std::string(e.what()) + "\"}", 
                               "application/json");
            }
        });
        std::cout << "\nCV Detection & Draft Reply Server starting on port 8080..." << std::endl;
        std::cout << "Endpoints:" << std::endl;
        std::cout << "  - GET  /health" << std::endl;
        std::cout << "  - POST /ai/inbox/detect-cv" << std::endl;
        std::cout << "  - POST /ai/inbox/draft-reply" << std::endl;
        std::cout << "  - POST /ai/inbox/classify" << std::endl;
        svr.listen("0.0.0.0", 8080);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}