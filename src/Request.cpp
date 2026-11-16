#include "Request.h"
#include "Model.h"
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

