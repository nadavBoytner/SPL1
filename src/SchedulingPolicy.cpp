#include "SchedulingPolicy.h"
#include "Model.h"
#include <algorithm>  
// clone
SchedulingPolicy* ArrivalOrderPolicy::clone() const{
    return new ArrivalOrderPolicy(*this);
}

SchedulingPolicy* ShortestJobPolicy::clone() const{
    return new ShortestJobPolicy(*this);
}

SchedulingPolicy* LongestPrefixPolicy::clone() const{
    return new LongestPrefixPolicy(*this);
}

//choose
std::vector<int> ArrivalOrderPolicy::choose(const std::vector<const Request*>& eligible, const std::vector<GPU>& gpus) const{
    std::vector<int> order;
    for(std::size_t i = 0;i < eligible.size();i++){
        order.push_back(static_cast<int>(i));
    }

    std::sort(order.begin(),order.end(),
        [&eligible](int a, int b){
            const Request* ra = eligible[a];
            const Request* rb = eligible[b];
            if(ra->getAdmissionCycle()==rb->getAdmissionCycle()){
                return ra->getId() < rb -> getId();
            }
            return ra->getAdmissionCycle() < rb->getAdmissionCycle();
        });
    return order;
}

std::vector<int> ShortestJobPolicy::choose(const std::vector<const Request*>& eligible, const std::vector<GPU>& gpus) const{
    std::vector<int> order;
    std::vector<std::size_t> blocks;
    for(std::size_t i = 0;i < eligible.size();i++){
        order.push_back(static_cast<int>(i));
        blocks.push_back(gpus[0].getBlocksForMemory(eligible[i]->getMaxNeededMemory()));
    }
    std::sort(order.begin(),order.end(),
        [&eligible, &blocks](int a, int b){
            if(blocks[a] == blocks[b]){
                return  eligible[a]->getId() < eligible[b] -> getId();
            }
            return blocks[a]<blocks[b];
        });
    return order;
}
std::vector<int> LongestPrefixPolicy::choose(const std::vector<const Request*>& eligible, const std::vector<GPU>& gpus) const{
    std::vector<int> order;
    std::vector<std::size_t> prefixlen;
    for(std::size_t i = 0;i < eligible.size();i++){
        order.push_back(static_cast<int>(i));
        size_t h =0;
        const Model& model = eligible[i]->getModel();
        std::size_t e = model.getEmbeddingSize();
        std::string id = model.getModeId();
        std::vector<std::string> token = eligible[i]->getCacheableTokens();
        for(std::size_t j = 0;j < gpus.size();j++){
            h = std::max(h, gpus[j].findLongestCachedPrefix(id,e,token).matchedTokens);
        }
        prefixlen.push_back(h);
    }
    std::sort(order.begin(),order.end(),
        [&eligible, &prefixlen](int a, int b){
            if(prefixlen[a] == prefixlen[b]){
                return  eligible[a]->getId() < eligible[b] -> getId();
            }
            return prefixlen[a]>prefixlen[b];
        });
    return order;
}

