#include "Request.h"
#include "Model.h"
#include "GPU.h"
//constructers

Request::Request(int id, const Model& model, int admissionCycle, std::string prompt):
    id(id),model(model),admissionCycle(admissionCycle),tokens(model.tokenizer(prompt)), 
    memoryBlocks(),reservedBlocks(0){}

GenerativeRequest::GenerativeRequest(int id, const Model& model, int admissionCycle, std::string prompt, std::string scriptedOutput,
    std::size_t maxOutputTokens): Request(id, model,admissionCycle, prompt), 
    scriptedOutput(scriptedOutput), maxOutputTokens(maxOutputTokens),
    outputTokens(model.tokenizer(scriptedOutput)), position(0), allocatedBlocks(0){
        if(outputTokens.size()>maxOutputTokens){
            outputTokens.resize(maxOutputTokens);
        }
    }

EmbeddingRequest::EmbeddingRequest(int id, const Model& model, int admissionCycle, std::string prompt): 
    Request(id, model,admissionCycle, prompt), processedTokens(0),result() {}

// copy contructers - help to clone
Request::Request(const Request& other, const Model& newModel) 
    : id(other.id), model(newModel), admissionCycle(other.admissionCycle),
    tokens(other.tokens), memoryBlocks(other.memoryBlocks), reservedBlocks(other.reservedBlocks){}

GenerativeRequest::GenerativeRequest(const GenerativeRequest& other, const Model& newModel):
    Request(other, newModel), scriptedOutput(other.scriptedOutput),
    maxOutputTokens(other.maxOutputTokens),outputTokens(other.outputTokens),
    position(other.position),allocatedBlocks(other.allocatedBlocks){}

EmbeddingRequest::EmbeddingRequest(const EmbeddingRequest& other, const Model& newModel):
    Request(other, newModel), processedTokens(other.processedTokens),
    result(other.result){}

//clone
Request* GenerativeRequest::clone(const Model& model) const{
    return new GenerativeRequest(*this, model);
}

Request* EmbeddingRequest::clone(const Model& model) const{
    return new EmbeddingRequest(*this, model);
}


//distructure

Request::~Request(){}

//geters
int Request::getId() const {return id;}
const Model& Request::getModel() const {return model;}
int Request::getAdmissionCycle() const {return admissionCycle;}

void Request::deallocateMemory(GPU& gpu){
    gpu.unreserve(getNeedToUnreserve(),*this);
    for(size_t i= 0; i< memoryBlocks.size(); i++)
    {
        std::size_t slot = memoryBlocks[i];
        gpu.release(slot);
    }
    memoryBlocks.clear();
}
// Genterative

std::vector<std::string> GenerativeRequest::getCacheableTokens() const {return tokens;}
std::size_t GenerativeRequest::getMaxNeededMemory() const{
    std::size_t e = model.getEmbeddingSize();
    return e * (tokens.size() + maxOutputTokens);
}
bool GenerativeRequest::finished() const{
    return tokens.size() + outputTokens.size() == position;
}
std::string GenerativeRequest::getOutput() const{
    std::string res="";
    for(size_t i = 0; i < outputTokens.size(); i++){
        res += outputTokens[i];
        if(i + 1< outputTokens.size()){
            res += " ";
        }
    }
    return res;
}
std::size_t GenerativeRequest::getNeedToUnreserve() const{
    return reservedBlocks - allocatedBlocks;
}  
void GenerativeRequest::allocateOneBlock(GPU& gpu, std::size_t slot) const{
    gpu.addMemoryBlock(*this, slot);
}
std::size_t GenerativeRequest::getNeededMemory(const GPU& gpu) const{
    std::size_t e = model.getEmbeddingSize();
    std::size_t n = tokens.size();
    std::size_t m = maxOutputTokens;
    std::size_t h = gpu.findLongestCachedPrefix(model.getModeId(),e,tokens).matchedTokens;
    return e * (n - h + m);
}
void GenerativeRequest::reserveMemory(GPU& gpu){
    std::size_t e = model.getEmbeddingSize();
    PrefixMatch match = gpu.findLongestCachedPrefix(model.getModeId(),e,tokens);
    for(size_t i = 0; i< match.blockIds.size();i++){
        gpu.acquire(match.blockIds[i]);
        memoryBlocks.push_back(match.blockIds[i]);
    }
    position = match.matchedTokens;
    reservedBlocks = gpu.getBlocksForMemory(getNeededMemory(gpu));
    gpu.reserve(reservedBlocks, *this);
}
std::vector<std::string> GenerativeRequest::sequenceUpTo(std::size_t p) const{
    std::vector<std::string> seq;
    std::size_t n = tokens.size();
    for(size_t i=0; i< p+1;i++){
        if(i<n){
            seq.push_back(tokens[i]);
        }else{
            seq.push_back(outputTokens[i-n]);
        }
    }
    return seq;
}
void GenerativeRequest::act(GPU& gpu){
    if(finished()){
        return;
    }
    std::size_t slot;
    size_t e = model.getEmbeddingSize();
    size_t t = gpu.getBlockSize()/model.getEmbeddingSize();
    if(position % t == 0){
        slot = gpu.allocateBlock(*this);
        memoryBlocks.push_back(slot);
        allocatedBlocks++;
    }
    slot = memoryBlocks.back();
    std::size_t offset = (position%t)*e;
    std::vector<std::string> seq = sequenceUpTo(position);
    model.generateTokenEmbedding(seq,gpu,slot,offset);
    if((position+1)%t==0){
        gpu.assignPrefixToBlock(slot,seq,model.getModeId());
    }
    position++;
}
//Embedding
std::vector<std::string> EmbeddingRequest::getCacheableTokens() const { return std::vector<std::string>();}
std::size_t EmbeddingRequest::getMaxNeededMemory() const{
    return model.getEmbeddingSize() * 2;
}
bool EmbeddingRequest::finished() const{
    return processedTokens == tokens.size();
}
std::string EmbeddingRequest::getOutput() const{
    std::string res="[";
    for(size_t i = 0; i < result.size(); i++){
        res += std::to_string(result[i]);
        if(i+1<result.size()){
            res+=", ";
        }
    }
    res+="]";
    return res;
}
std::size_t EmbeddingRequest::getNeedToUnreserve() const{
    return reservedBlocks - memoryBlocks.size();
}
void EmbeddingRequest::allocateOneBlock(GPU& gpu, std::size_t slot) const{
    gpu.addMemoryBlock(*this, slot);
}
std::size_t EmbeddingRequest::getNeededMemory(const GPU& gpu) const{
    return 2 * model.getEmbeddingSize();
}
void EmbeddingRequest::reserveMemory(GPU& gpu){
    std::size_t e = model.getEmbeddingSize();
    reservedBlocks = gpu.getBlocksForMemory(2*e);
    gpu.reserve(reservedBlocks, *this);
    for(size_t i = 0; i<reservedBlocks; i++){
        std::size_t slot = gpu.allocateBlock(*this);
        memoryBlocks.push_back(slot);
    }
}
std::size_t EmbeddingRequest::slotOf(std::size_t index, std::size_t blockSize) const{
    return memoryBlocks[index / blockSize];
}
std::size_t EmbeddingRequest::offsetOf(std::size_t index, std::size_t blockSize) const{
    return index % blockSize;
}
void EmbeddingRequest::act(GPU& gpu){
     if(finished()){
        return;
    }
    std::size_t B = gpu.getBlockSize();
    size_t e = model.getEmbeddingSize();

    std::vector<std::string> current;
    current.push_back(tokens[processedTokens]);
    model.generateTokenEmbedding(current,gpu,slotOf(e,B),offsetOf(e,B));
    for(std::size_t j = 0; j < e;j++){//update result
        int curr = gpu.readFromBlock(slotOf(e+j,B),offsetOf(e+j,B));
        int sum = gpu.readFromBlock(slotOf(j,B),offsetOf(j,B));
        gpu.writeToBlock(slotOf(j,B),offsetOf(j,B),sum+curr);
    }
    processedTokens++;

    //if was last token
    if(finished()){
        int n = static_cast<int>(tokens.size());
        for (std::size_t j = 0; j < e;j++){//calcukate the avrage
            int sum = gpu.readFromBlock(slotOf(j,B),offsetOf(j,B));
            result.push_back(sum / n);
        }
    }
}
