# include "MemoryBlock.h"

MemoryBlock::MemoryBlock(size_t size):
    size(size), refCount(0), data(new int[size]()){}

MemoryBlock::~MemoryBlock(){
    delete[] data;
}

MemoryBlock::MemoryBlock(const MemoryBlock& other): 
    size(other.size), refCount(0), data(new int[other.size]){
        for (size_t i = 0; i < size; i++)
        {
            data[i] = other.data[i];
        }
    }

MemoryBlock::MemoryBlock(MemoryBlock&& other): 
    size(other.size), refCount(0), data(other.data){
        other.data = nullptr;
        other.size = 0;
        other.refCount = 0;
    }

MemoryBlock& MemoryBlock::operator=(const MemoryBlock& other){
    if(other!=this){

    }
    return *this
}