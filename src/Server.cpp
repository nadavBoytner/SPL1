#include <fstream>
#include <stdexcept>
#include "json.hpp"
#include "Model.h"
#include "Server.h"
#include "SchedulingPolicy.h"

using namespace std;
using json = nlohmann::json;

Config Server::readConfig(string configPath) const{
    std::ifstream input(configPath);
    json config;
    input >> config;
    // All the GPUs have the same block size.
    size_t blockSize = config["block_size"].get<size_t>();
    vector<GPUConfig> gpuConfigs;
    for(auto& gpu: config["gpus"]){
        size_t memoryBytes = gpu["memory_bytes"].get<size_t>();
        GPUConfig gpuConfig{memoryBytes};
        gpuConfigs.push_back(gpuConfig);
    }
    vector<ModelConfig> modelConfigs;
    for(auto& model: config["models"]){
        string id = model["id"].get<string>();
        size_t embeddingSize = model["embedding_size"].get<size_t>();
        ModelConfig modelConfig{id, embeddingSize};
        modelConfigs.push_back(modelConfig);
    }
    vector<RequestConfig> requestConfigs;
    for(auto& request: config["requests"]){
        size_t id = request["id"].get<size_t>();
        string typeStr = request["type"].get<string>();
        if(typeStr != "generative" && typeStr != "text_embedder"){
            throw invalid_argument("invalid request type");
        }
        RequestType type = typeStr=="generative"?RequestType::Generative:RequestType::TextEmbedder;
        string modelId = request["model"].get<string>();
        int admissionCycle = request["cycle"].get<int>();
        string prompt = request["prompt"].get<string>();
        string scriptedOutput = "";
        if (request.count("scripted_output") > 0) {
            scriptedOutput = request["scripted_output"].get<string>();
        }
        size_t maxOutputTokens = 0;
        if (request.count("max_output_tokens") > 0) {
            maxOutputTokens = request["max_output_tokens"].get<size_t>();
        }
        RequestConfig reqConfig{id, type, modelId, admissionCycle, prompt, scriptedOutput, maxOutputTokens};
        requestConfigs.push_back(reqConfig);
    }

    if (config.count("scheduling_policy") == 0) {
        throw invalid_argument("missing scheduling policy");
    }
    string policy = config["scheduling_policy"].get<string>();

    return Config{blockSize, gpuConfigs, modelConfigs, requestConfigs, policy};
}

Server::Server(std::string configPath)
    : gpus(),
      models(),
      nonSubmittedRequests(),
      waitingRequests(),
      activeRequests(),
      policy(nullptr),
      cycle(0),
      activeGpus() {
    Config config = readConfig(configPath);
    for (std::size_t i = 0; i < config.gpus.size(); i++) {
        std::size_t blocksNum = config.gpus[i].memoryBytes / config.blockSize;
        gpus.push_back(GPU(blocksNum, config.blockSize));
    }
    for (std::size_t i = 0; i < config.models.size(); i++) {
        models.push_back(new Model(config.models[i].id, config.models[i].embeddingSize));
    }
    for (std::size_t i = 0; i < config.requests.size(); i++) {
        const RequestConfig& rc = config.requests[i];
        Model* model = findModel(rc.modelId);
        int id = static_cast<int>(rc.id);
        if (rc.type == RequestType::Generative) {
            nonSubmittedRequests.push_back(
                new GenerativeRequest(id, *model, rc.admissionCycle, rc.prompt,
                                      rc.scriptedOutput, rc.maxOutputTokens));
        } else {
            nonSubmittedRequests.push_back(
                new EmbeddingRequest(id, *model, rc.admissionCycle, rc.prompt));
        }
    }
    if (config.policy == "arrivalOrder") {
        policy = new ArrivalOrderPolicy();
    } else if (config.policy == "shortestJob") {
        policy = new ShortestJobPolicy();
    } else if (config.policy == "longestPrefix") {
        policy = new LongestPrefixPolicy();
    } else {
        throw std::runtime_error("Server: unknown scheduling policy");
    }
}

void Server::clear(){
    for(std::size_t i =0; i<models.size(); i++){
        delete models[i];
    }
    for (std::size_t i = 0; i < nonSubmittedRequests.size(); i++) {
        delete nonSubmittedRequests[i];
    }
    for (std::size_t i = 0; i < waitingRequests.size(); i++) {
        delete waitingRequests[i];
    }
    for (std::size_t i = 0; i < activeRequests.size(); i++) {
        delete activeRequests[i];
    }
    delete policy;
    nonSubmittedRequests.clear();
    waitingRequests.clear();
    activeRequests.clear();
    models.clear();
    gpus.clear();
    activeGpus.clear();
    policy = nullptr;
    cycle = 0;
}

Server::~Server(){
    clear();
}

Model* Server::findModel(const std::string& id) const {
    for (std::size_t i = 0; i < models.size(); i++) {
        if (models[i]->getModeId() == id) {
            return models[i];
        }
    }
    throw std::runtime_error("findModel: model not found");
}

void Server::copyFrom(const Server& other){
    gpus = other.gpus; 
    for (std::size_t i = 0; i < other.models.size(); i++) {
    models.push_back(new Model(*other.models[i]));
}
for (std::size_t i = 0; i < other.nonSubmittedRequests.size(); i++) {
    nonSubmittedRequests.push_back(other.nonSubmittedRequests[i]->clone(*findModel(other.nonSubmittedRequests[i]->getModel().getModeId())));
}
for(std::size_t i = 0; i < other.waitingRequests.size(); i++){
    Request* r = other.waitingRequests[i];
        waitingRequests.push_back(r->clone(*findModel(r->getModel().getModeId())));
}
for (std::size_t i = 0; i < other.activeRequests.size(); i++) {
        Request* r = other.activeRequests[i];
        activeRequests.push_back(r->clone(*findModel(r->getModel().getModeId())));
}
policy=nullptr;
if(other.policy != nullptr){
    policy = other.policy->clone();
}
cycle = other.cycle;
activeGpus = other.activeGpus;
}

Server::Server(const Server& other):
    gpus(),
    models(),
    nonSubmittedRequests(),
    waitingRequests(),
    activeRequests(),
    policy(nullptr),
    cycle(0),
    activeGpus() {
    copyFrom(other);
}

Server::Server(Server&& other)
    : gpus(),
      models(),
      nonSubmittedRequests(),
      waitingRequests(),
      activeRequests(),
      policy(other.policy),
      cycle(other.cycle),
      activeGpus() {
    gpus.swap(other.gpus);
    models.swap(other.models);
    nonSubmittedRequests.swap(other.nonSubmittedRequests);
    waitingRequests.swap(other.waitingRequests);
    activeRequests.swap(other.activeRequests);
    activeGpus.swap(other.activeGpus);
    other.policy = nullptr;
    other.cycle = 0;
}

Server& Server::operator=(const Server& other){
    if(this == &other){
        return *this;
    }
    clear();
    copyFrom(other);
    return *this;
}

Server& Server::operator=(Server&& other) {
    if (this == &other) {
        return *this;
    }
    clear();            
    gpus.swap(other.gpus);
    models.swap(other.models);
    nonSubmittedRequests.swap(other.nonSubmittedRequests);
    waitingRequests.swap(other.waitingRequests);
    activeRequests.swap(other.activeRequests);
    activeGpus.swap(other.activeGpus);
    policy = other.policy;
    cycle = other.cycle;
    other.policy = nullptr;
    other.cycle = 0;
    return *this;
}