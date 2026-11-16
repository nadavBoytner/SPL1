#ifndef REQUEST_H_
#define REQUEST_H_

#include <string>
#include <vector>

class Model;
class GPU;

class Request{
protected:
    int id;
    const Model& model;
    int admissionCycle;
    std::vector<std::string> tokens;
    std::vector<std::size_t> memoryBlocks;
    std::size_t reservedBlocks;
public:
    Request(int id, const Model& model, int admissionCycle, std::string prompt);
    Request(const Request& other, const Model& newModel);
    virtual Request* clone(const Model& model) const = 0;
    virtual std::size_t getNeededMemory(const GPU& gpu) const=0;
    virtual std::size_t getMaxNeededMemory() const=0;
    virtual std::vector<std::string> getCacheableTokens() const=0;
    virtual void allocateOneBlock(GPU& gpu, std::size_t slot) const = 0;
    virtual Request* clone(const Model& model) const = 0;
    void deallocateMemory(GPU& gpu);
    virtual void act(GPU& gpu)=0;
    virtual bool finished() const=0;
    virtual void reserveMemory(GPU& gpu) = 0;
    virtual std::string getOutput() const = 0;
    virtual std::size_t getNeedToUnreserve() const = 0;
    int getId() const;
    const Model& getModel() const;
    int getAdmissionCycle() const;
    virtual ~Request();
};

class GenerativeRequest: public Request{
private:
    std::string scriptedOutput;
    std::size_t maxOutputTokens;
    
public:
    GenerativeRequest(int id, const Model& model, int admissionCycle, std::string prompt, std::string scriptedOutput,
    std::size_t maxOutputTokens);
    GenerativeRequest(const GenerativeRequest& other, const Model& newModel);
    virtual std::size_t getNeededMemory(const GPU& gpu) const;
    virtual std::size_t getMaxNeededMemory() const;
    virtual std::vector<std::string> getCacheableTokens() const;
    virtual void allocateOneBlock(GPU& gpu, std::size_t slot) const;
    virtual Request* clone(const Model& model) const;
    virtual void act(GPU& gpu);
    virtual bool finished() const;
    virtual void reserveMemory(GPU& gpu);
    virtual std::string getOutput() const;
    virtual std::size_t getNeedToUnreserve() const;
};

class EmbeddingRequest: public Request{
public:
    EmbeddingRequest(int id, const Model& model, int admissionCycle, std::string prompt);
    EmbeddingRequest(const EmbeddingRequest& other, const Model& newModel);
    virtual std::size_t getNeededMemory(const GPU& gpu) const;
    virtual std::size_t getMaxNeededMemory() const;
    virtual std::vector<std::string> getCacheableTokens() const;
    virtual void allocateOneBlock(GPU& gpu, std::size_t slot) const;
    virtual Request* clone(const Model& model) const ;
    virtual void act(GPU& gpu);
    virtual bool finished() const;
    virtual void reserveMemory(GPU& gpu);
    virtual std::string getOutput() const;
    virtual std::size_t getNeedToUnreserve() const;
};

#endif