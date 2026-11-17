#include <fstream>
#include <stdexcept>
#include "json.hpp"
#include <iostream>
#include "Server.h"
#include "Model.h"
#include "SchedulingPolicy.h"
#include "GPU.h"

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

void Server::serve(){
    while(step()){}
    std::size_t cacheHits = 0;
    std::size_t evictions = 0;
    for(size_t i = 0; i < gpus.size();i++){
        cacheHits += gpus[i].getCacheHits();
        evictions += gpus[i].getEvictions();
    }
    std::cout << "summary: cycles="<<cycle<<" cache_hits="<<cacheHits<<" evictions="<<evictions << std::endl;
}
bool Server::step(){
    //check if finished
    if(activeRequests.size() == 0 && waitingRequests.size() == 0 && nonSubmittedRequests.size() == 0){
        return false;
    }
    //submutted to waiting or rejected
    std::vector<Request*> stillPendingRequest;
    std::size_t blocks;
    for (size_t i = 0; i < nonSubmittedRequests.size(); i++){
        Request* r = nonSubmittedRequests[i];
        if(r -> getAdmissionCycle() == cycle){
                blocks= gpus[0].getBlocksForMemory(r->getMaxNeededMemory());
                bool flag = false;
                for (size_t j =0;j < gpus.size() && !flag; j++){
                    if(blocks <= gpus[j].getBlocksNum()){//accepted
                        flag = true;
                        waitingRequests.push_back(r);
                    }
                }
                if(!flag){
                    std::cout <<"rejected "<< r->getId() <<" cycle="<<cycle<< std::endl; 
                    delete r;
                }
        }
        else{
            stillPendingRequest.push_back(r);
        }
    }
    nonSubmittedRequests = stillPendingRequest;
    //
    std::vector<const Request*> eligible;
    for (size_t i = 0; i < waitingRequests.size(); i++)
    {
        eligible.push_back(waitingRequests[i]);
    }
    std::vector<int> seq = policy->choose(eligible,gpus);
//combines request to GPU
    std::vector<Request*> stillAwait;
    for(size_t i = 0; i < eligible.size(); i++){//each request
        Request* r = waitingRequests[seq[i]];
        Model model = r->getModel();
        std::string modelId = r->getModel().getModeId();
        std::vector<std::string> tokens = r->getCacheableTokens();
        std::size_t e = r->getModel().getEmbeddingSize();
        int bestIdx = -1;
        std::size_t bestCached;
        std::size_t bestNeed;
        PrefixMatch bestmatch;
        for(size_t g = 0; g < gpus.size(); g++){//each gpu
            PrefixMatch match = gpus[g].findLongestCachedPrefix(modelId,e ,tokens);
            std::size_t cached = match.blockIds.size();
            std::size_t newBlocks = gpus[g].getBlocksForMemory(r->getNeededMemory(gpus[g])); 
            std::size_t idleCached = 0;
            std::vector<std::size_t> blockIds =match.blockIds ;
            for(size_t s =0 ; s < blockIds.size();s++){
                if(gpus[g].isIdle(blockIds[s])){
                    idleCached++;
                }
            }
            std::size_t need = newBlocks + idleCached;
            if(need > gpus[g].availableSlots()){
                continue;
            }
            if(bestIdx == -1 || cached > bestCached ||
                (cached == bestCached && bestNeed > need)
            ){
                bestCached = cached;
                bestNeed =need;
                bestIdx = g;
                bestmatch = match;
            }
            
        }
        if(bestIdx == -1){//not fit in any gpu.
            stillAwait.push_back(r);
            continue;
        }
        int src = -1;
        std::size_t srcTokens = bestmatch.matchedTokens;     // צריך לנצח את זה
        PrefixMatch srcMatch;

        for (std::size_t j = 0; j < gpus.size(); j++) {
            if (static_cast<int>(j) == bestIdx) {
                continue;                                       
            }
            PrefixMatch m = gpus[j].findLongestCachedPrefix(modelId, e, tokens);
            if (m.matchedTokens > srcTokens) {                  
                src = j;
                srcTokens = m.matchedTokens;
                srcMatch = m;
            }
        }

        if (src != -1) {
            gpus[bestIdx].copyMemoryFromOtherGPU(gpus[src], srcMatch.blockIds);
        }//copy ot the biggest prefix

        std::size_t hits = gpus[bestIdx].findLongestCachedPrefix(modelId, e, tokens).blockIds.size();
        r->reserveMemory(gpus[bestIdx]);
        gpus[bestIdx].addCacheHits(hits);
        activeRequests.push_back(r);
        activeGpus.push_back(static_cast<std::size_t>(bestIdx));

        
   
    }
    waitingRequests = stillAwait;

    std::vector<Request*> stillActive;
    std::vector<std::size_t> stillGpus;

    for (std::size_t i = 0; i < activeRequests.size(); i++) {//one token push to each request
        Request* r = activeRequests[i];
        std::size_t g = activeGpus[i];

        r->act(gpus[g]);                 

        if (r->finished()) {
            std::cout << "finished " << r->getId()
                      << " cycle=" << cycle
                      << " gpu=" << g
                      << ": " << r->getOutput() << std::endl;
            r->deallocateMemory(gpus[g]);     
            delete r;
        } else {
            stillActive.push_back(r);
            stillGpus.push_back(g);
        }
    }

    activeRequests = stillActive;
    activeGpus = stillGpus;

    cycle++;
    return true;
}
