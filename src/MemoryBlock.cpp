# include "MemoryBlock.h"
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
