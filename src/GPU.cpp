#include <stdexcept>
#include "GPU.h"
#include "MemoryBlock.h"
#include "Request.h"

//constructor
GPU::GPU(std::size_t blockNum, std::size_t blockSize)
: blocksNum(blockNum), 
blockSize(blockSize),
memory(blockNum, nullptr),
slotModel(blockNum),
slotKey(blockNum),
totalReserved(0),
reservations(),
evictions(0),
cacheHits(0)
{}

void GPU::clear(){
    for(std::size_t i =0; i<memory.size(); i++){
        delete memory[i];
        memory[i]=nullptr;
    }
}

//Distructor
GPU::~GPU(){
    clear();
}

//Copy constructor
GPU::GPU(const GPU& other):
blocksNum(other.blocksNum),
blockSize(other.blockSize),
memory(other.memory.size(), nullptr),
slotModel(other.slotModel),
slotKey(other.slotKey),
totalReserved(other.totalReserved),
reservations(other.reservations),
evictions(other.evictions),
cacheHits(other.cacheHits){
    for(std::size_t i=0; i<other.memory.size(); i++){
        if(other.memory[i]!=nullptr){
            memory[i]=other.memory[i]->clone();
            for(std::size_t j=0; j<other.memory[i]->getRefCount(); j++){
                memory[i]->acquire();
            }
        }
    }
}

GPU& GPU::operator=(const GPU& other){
if(this == &other){
    return *this;
}
clear();
blocksNum=other.blocksNum;
blockSize=other.blockSize;
memory.assign(other.memory.size(), nullptr);
slotModel=other.slotModel;
slotKey=other.slotKey;
totalReserved=other.totalReserved;
reservations=other.reservations;
evictions=other.evictions;
cacheHits=other.cacheHits;
    for(std::size_t i=0; i<other.memory.size(); i++){
        if(other.memory[i]!=nullptr){
            memory[i]=other.memory[i]->clone();
            for(std::size_t j=0; j<other.memory[i]->getRefCount(); j++){
                memory[i]->acquire();
            }
        }
    }
    return *this;
}

GPU::GPU(GPU&& other)
    : blocksNum(other.blocksNum),
      blockSize(other.blockSize),
      memory(),               
      slotModel(),
      slotKey(),
      totalReserved(other.totalReserved),
      reservations(),
      evictions(other.evictions),
      cacheHits(other.cacheHits) {
    memory.swap(other.memory);
    slotModel.swap(other.slotModel);
    slotKey.swap(other.slotKey);
    reservations.swap(other.reservations);
    other.blocksNum = 0;
    other.totalReserved = 0;
    other.evictions = 0;
    other.cacheHits = 0;
}

GPU& GPU::operator=(GPU&& other) {
    if (this == &other) {
        return *this;
    }
    clear();                          
    blocksNum = other.blocksNum;
    blockSize = other.blockSize;
    totalReserved = other.totalReserved;
    evictions = other.evictions;
    cacheHits = other.cacheHits;
    memory.swap(other.memory);
    slotModel.swap(other.slotModel);
    slotKey.swap(other.slotKey);
    reservations.swap(other.reservations);
    other.memory.clear();
    other.slotModel.clear();
    other.slotKey.clear();
    other.reservations.clear();
    other.blocksNum = 0;
    other.totalReserved = 0;
    other.evictions = 0;
    other.cacheHits = 0;
    return *this;
}

std::size_t GPU::getBlocksForMemory(std::size_t memory) const{
    
    if(memory % blockSize ==0){
        return memory/ blockSize;
    }
    else{
        return memory/ blockSize+1;
    }
}

int GPU::readFromBlock(std::size_t blockId, std::size_t offset) const{
     if (blockId >= memory.size() || memory[blockId] == nullptr) {
        throw std::runtime_error("readFromBlock: slot is empty or out of range");
        }
    return memory[blockId]->readFromBlock(offset);
}

void GPU::writeToBlock(std::size_t blockId, std::size_t offset, int value){
    if (blockId >= memory.size() || memory[blockId] == nullptr) {
        throw std::runtime_error("writeToBlock: slot is empty or out of range");
    }
    memory[blockId]->writeToBlock(offset,value);
}

void GPU::acquire(std::size_t slot){
    if (slot >= memory.size() || memory[slot] == nullptr) {
        throw std::runtime_error("acquire: slot is empty or out of range");
    }
    memory[slot]->acquire();
}

void GPU::release(std::size_t slot){
     if (slot >= memory.size() || memory[slot] == nullptr) {
        throw std::runtime_error("release: slot is empty or out of range");
    }
    IdleAction action = memory[slot]->release();
    if(memory[slot]->getRefCount()>0){
        return;
    }
    if(action==IdleAction::Destroy || slotKey[slot].empty()){
        delete memory[slot];
        memory[slot] =nullptr;
        slotKey[slot].clear();
        slotModel[slot].clear();
    }
}

void GPU::reserve(std::size_t n, const Request& request){
    totalReserved+=n;
    reservations[request.getId()]+=n;
}

void GPU::unreserve(std::size_t n, const Request& request){
    if(n==0){
        return;
    }
    int id = request.getId();
    auto it = reservations.find(id);
    if(it == reservations.end() || n>totalReserved || n>reservations[id]){
          throw std::runtime_error("unreserve: more than reserved");
    }
    reservations[id]-=n;
    totalReserved-=n;
    if(reservations[id]==0){
        reservations.erase(id);
    }

}

std::size_t GPU::availableSlots() const{
    std::size_t count = 0;
    for(MemoryBlock* block : memory){
        if(block!=nullptr && block->getRefCount()>0){
            count++;
        }
    }
    if(count + totalReserved>blocksNum){
        throw std::runtime_error("availableSlots: overflow");
    }
    return blocksNum - count - totalReserved;
}

void GPU::addMemoryBlock(const GenerativeRequest& request, std::size_t slot) {
    (void)request;  
    if (slot >= memory.size() || memory[slot] != nullptr) {
        throw std::runtime_error("addMemoryBlock: slot out of range or not empty");
    }
    memory[slot] = new KVBlock(blockSize);
    slotKey[slot].clear();     
    slotModel[slot].clear();
}

void GPU::addMemoryBlock(const EmbeddingRequest& request, std::size_t slot) {
    (void)request;
    if (slot >= memory.size() || memory[slot] != nullptr) {
        throw std::runtime_error("addMemoryBlock: slot out of range or not empty");
    }
    memory[slot] = new EmbeddingWorkspace(blockSize);
    slotKey[slot].clear();
    slotModel[slot].clear();
}

std::size_t GPU::allocateBlock(const Request& request){
    if(reservations.find(request.getId())==reservations.end() || reservations[request.getId()]==0){
        throw std::runtime_error("allocateBlock: you can not allocate block without an order");
    }
    std::size_t index=0;
    while(index<blocksNum && memory[index]!=nullptr){
        index+=1;
    }
    if(index==blocksNum){
        index=0;
        while(index<blocksNum && memory[index]!=nullptr && memory[index]->getRefCount()!=0){
            index+=1;
        }
        if(index==blocksNum){
            throw std::runtime_error("allocateBlock: there isn't available blocks");
        }
        else{
            delete memory[index];
            memory[index] =nullptr;
            slotKey[index].clear();
            slotModel[index].clear();
            evictions++;
        }
    }
    request.allocateOneBlock(*this, index);
    unreserve(1, request);
    acquire(index);
    return index;
}

void GPU::assignPrefixToBlock(std::size_t block, const std::vector<std::string>& prefixTokens, const std::string& modelId){
    if(block>=blocksNum || memory[block]==nullptr){
        throw std::runtime_error("assignPrefixToBlock: block is empty or out of range");
    }
    for (std::size_t i = 0; i < blocksNum; i++) {
        if (i == block) {
            continue;
        }
        if (slotModel[i] == modelId && slotKey[i] == prefixTokens) {
            return;
        }
    }
        slotKey[block] = prefixTokens;
        slotModel[block] = modelId;
    
}

std::size_t GPU::getCacheHits() const {
    return cacheHits;
}

std::size_t GPU::getEvictions() const {
    return evictions;
}

std::size_t GPU::getBlockSize() const {
    return blockSize;
}

std::size_t GPU::getBlocksNum() const {
    return blocksNum;
}

void GPU::addCacheHits(std::size_t n) {
    cacheHits += n;
}

PrefixMatch GPU::findLongestCachedPrefix(const std::string& modelId, std::size_t embeddingSize, const std::vector<std::string>& promptTokens) const{
    PrefixMatch res = {0, std::vector<std::size_t>()};
    res.matchedTokens =0;
    std::size_t t = blockSize / embeddingSize;
    std::size_t covered = t;
    while (covered<=promptTokens.size()){
        std::vector<std::string> key(promptTokens.begin(), promptTokens.begin() + covered);
        bool found = false;
        for (std::size_t i = 0; i < blocksNum; i++) {
            if (memory[i] != nullptr && slotModel[i] == modelId && slotKey[i] == key) {
                res.blockIds.push_back(i);
                res.matchedTokens = covered;
                found = true;
                break;                          
            }
        }
        if (!found) {
            break;                              
        }
        covered += t;  
    }
    return res;
}

bool GPU::isInChain(const std::vector<std::size_t>& chain, std::size_t slot) {
    for (std::size_t i = 0; i < chain.size(); i++) {
        if (chain[i] == slot) {
            return true;
        }
    }
    return false;
}

void GPU::copyMemoryFromOtherGPU(const GPU& other, const std::vector<std::size_t>& blocks) {
    std::vector<std::size_t> chain; 
    for (std::size_t b = 0; b < blocks.size(); b++) {
        std::size_t s = blocks[b];                    
        if (s >= other.memory.size() || other.memory[s] == nullptr) {
            throw std::runtime_error("copyMemoryFromOtherGPU: source slot is empty or out of range");
        }
        const std::string& model = other.slotModel[s];
        const std::vector<std::string>& key = other.slotKey[s];
        bool exists = false;
        for (std::size_t i = 0; i < blocksNum; i++) {
            if (memory[i] != nullptr && slotModel[i] == model && slotKey[i] == key) {
                chain.push_back(i);                    
                exists = true;
                break;
            }
        }
        if (exists) {
            continue;                                   
        }
         std::size_t target = blocksNum;
        for (std::size_t i = 0; i < blocksNum; i++) {
            if (memory[i] == nullptr) {
                target = i;
                break;
            }
        }
         if (target == blocksNum) {
            for (std::size_t i = 0; i < blocksNum; i++) {
                if (memory[i]->getRefCount() == 0 && !isInChain(chain, i)) {
                    target = i;
                    break;
                }
            }
            if (target == blocksNum) {
                throw std::runtime_error("copyMemoryFromOtherGPU: no slot available");
            }
            delete memory[target];
            memory[target] = nullptr;
            slotKey[target].clear();
            slotModel[target].clear();
            evictions++;
        }
        memory[target] = other.memory[s]->clone();       // refCount 0, כלומר idle
        slotKey[target] = key;
        slotModel[target] = model;
        chain.push_back(target);
    }
}
bool GPU::isIdle(std::size_t slot) const {
    return memory[slot] != nullptr && memory[slot]->getRefCount() == 0;
}

