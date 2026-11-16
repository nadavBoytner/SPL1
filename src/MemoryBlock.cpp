# include "MemoryBlock.h"
//MemoryBlock
//constructer
MemoryBlock::MemoryBlock(size_t size):
    size(size), refCount(0), data(new int[size]()){}

    //disractur
MemoryBlock::~MemoryBlock(){
    delete[] data;
}
//copy contructer
MemoryBlock::MemoryBlock(const MemoryBlock& other): 
    size(other.size), refCount(0), data(new int[other.size]){
        for (size_t i = 0; i < size; i++)
        {
            data[i] = other.data[i];
        }
    }
//rvalue constructer
MemoryBlock::MemoryBlock(MemoryBlock&& other): 
    size(other.size), refCount(0), data(other.data){
        other.data = nullptr;
        other.size = 0;
        other.refCount = 0;
    }
//copy assignment
MemoryBlock& MemoryBlock::operator=(const MemoryBlock& other){
    if(&other != this){
        size = other.size;
        int* newData = new int[size];
        for (int i = 0; i < size; i++){
            newData[i] = other.data[i];
        }
        delete[] data;
        data = newData;
        size = other.size;
        refCount = 0;
    }
    return *this;
}

//rvalue assignment
MemoryBlock& MemoryBlock::operator=(MemoryBlock&& other){
    if(this != &other){
        delete[] data;
        data = other.data;
        refCount = 0;
        size = other.size;

        other.data = nullptr;
        other.size = 0;
        other.refCount = 0;
    }
    return *this;
}

int MemoryBlock::readFromBlock(std::size_t offset) const{
    if(offset>=size){
        std::__throw_out_of_range("readFromBlock: offset out of range");
    }
    return data[offset];
}

void MemoryBlock::writeToBlock(std::size_t offset, int value){
    if(offset>=size){
        std::__throw_out_of_range("writeToBlock: offset out of range");
    }
    data[offset] = value;
}

void MemoryBlock::acquire(){
    refCount++;
}

IdleAction MemoryBlock::release(){
    if(refCount > 0){
        refCount--;
    }
    if(refCount == 0){
        return onLastRelease();
    }
    return IdleAction::KeepCached;
}
//KVBlock
KVBlock::KVBlock(size_t size):MemoryBlock(size) {}
IdleAction KVBlock::onLastRelease() const{
    return IdleAction::KeepCached;
}
MemoryBlock* KVBlock::clone() const{
    return new KVBlock(*this);
}

//EmbeddingWorkspace
EmbeddingWorkspace::EmbeddingWorkspace(size_t size): MemoryBlock(size) {}
IdleAction EmbeddingWorkspace::onLastRelease() const{
    return IdleAction::Destroy;
}
MemoryBlock* EmbeddingWorkspace::clone() const{
    return new EmbeddingWorkspace(*this);
}