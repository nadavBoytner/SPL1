// ============================================================================
// GPU מזויף לבדיקות של Request בלבד.
// לא חלק מההגשה! מאפשר לבדוק את Request לפני שה-GPU האמיתי מוכן.
// מממש רק את מה ש-Request ו-Model משתמשים בו, בצורה הכי פשוטה:
//   - הקצאה ב-slot הריק הנמוך ביותר (בלי eviction)
//   - ספירת שמירות לפי id של בקשה
//   - מפתחות prefix ו-findLongestCachedPrefix כמו בסעיף 4.2
// כל המצב נשמר במבנה גלובלי כאן (לא בשדות של GPU), כדי לא לשנות את GPU.h.
// ============================================================================
#include "GPU.h"
#include "MemoryBlock.h"
#include "Request.h"
#include <map>
#include <stdexcept>

struct FakeState {
    std::size_t blockSize;
    std::vector<MemoryBlock*> slots;
    std::vector<int> refs;
    std::vector<bool> hasKey;
    std::vector<std::string> keyModel;
    std::vector<std::vector<std::string> > keyTokens;
    std::map<int, long> reservedById;
    long totalReserved;
    int violations;   // allocate בלי שמירה, release של slot ריק וכו'
    FakeState() : blockSize(0), slots(), refs(), hasKey(), keyModel(), keyTokens(),
                  reservedById(), totalReserved(0), violations(0) {}
};

static std::map<const GPU*, FakeState> states;

static FakeState& st(const GPU* g) { return states.at(g); }

static void destroySlot(FakeState& s, std::size_t slot) {
    delete s.slots[slot];
    s.slots[slot] = nullptr;
    s.refs[slot] = 0;
    s.hasKey[slot] = false;
    s.keyModel[slot].clear();
    s.keyTokens[slot].clear();
}

GPU::GPU(std::size_t blocksNum, std::size_t blockSize)
    : blocksNum(blocksNum), blockSize(blockSize), memory() {
    FakeState s;
    s.blockSize = blockSize;
    s.slots.assign(blocksNum, nullptr);
    s.refs.assign(blocksNum, 0);
    s.hasKey.assign(blocksNum, false);
    s.keyModel.assign(blocksNum, "");
    s.keyTokens.assign(blocksNum, std::vector<std::string>());
    s.totalReserved = 0;
    s.violations = 0;
    states[this] = s;
}

std::size_t GPU::getBlockSize() const { return blockSize; }

std::size_t GPU::getBlocksForMemory(std::size_t memory) const {
    return (memory + blockSize - 1) / blockSize;
}

void GPU::reserve(std::size_t n, const Request& request) {
    FakeState& s = st(this);
    s.reservedById[request.getId()] += static_cast<long>(n);
    s.totalReserved += static_cast<long>(n);
}

void GPU::unreserve(std::size_t n, const Request& request) {
    FakeState& s = st(this);
    s.reservedById[request.getId()] -= static_cast<long>(n);
    s.totalReserved -= static_cast<long>(n);
    if (s.reservedById[request.getId()] < 0) s.violations++;
}

std::size_t GPU::allocateBlock(const Request& request) {
    FakeState& s = st(this);
    if (s.reservedById[request.getId()] <= 0) s.violations++;   // הקצאה בלי שמירה
    std::size_t slot = 0;
    while (slot < s.slots.size() && s.slots[slot] != nullptr) slot++;
    if (slot == s.slots.size()) throw std::runtime_error("fake GPU full (enlarge it in the test)");
    request.allocateOneBlock(*this, slot);      // double dispatch
    acquire(slot);                              // הבקשה תופסת את הבלוק מיד
    s.reservedById[request.getId()]--;          // צורך בלוק אחד מהשמירה
    s.totalReserved--;
    return slot;
}

void GPU::addMemoryBlock(const GenerativeRequest&, std::size_t slot) {
    st(this).slots[slot] = new KVBlock(blockSize);
}

void GPU::addMemoryBlock(const EmbeddingRequest&, std::size_t slot) {
    st(this).slots[slot] = new EmbeddingWorkspace(blockSize);
}

void GPU::acquire(std::size_t slot) {
    FakeState& s = st(this);
    if (s.slots[slot] == nullptr) { s.violations++; return; }
    s.slots[slot]->acquire();
    s.refs[slot]++;
}

void GPU::release(std::size_t slot) {
    FakeState& s = st(this);
    if (s.slots[slot] == nullptr || s.refs[slot] == 0) { s.violations++; return; }
    IdleAction action = s.slots[slot]->release();
    s.refs[slot]--;
    if (s.refs[slot] == 0 && (action == IdleAction::Destroy || !s.hasKey[slot])) {
        destroySlot(s, slot);
    }
}

void GPU::assignPrefixToBlock(std::size_t block, const std::vector<std::string>& prefixTokens,
                              const std::string& modelId) {
    FakeState& s = st(this);
    for (std::size_t i = 0; i < s.slots.size(); i++) {
        if (s.hasKey[i] && s.keyModel[i] == modelId && s.keyTokens[i] == prefixTokens) return; // כפילות
    }
    s.hasKey[block] = true;
    s.keyModel[block] = modelId;
    s.keyTokens[block] = prefixTokens;
}

int GPU::readFromBlock(std::size_t blockId, std::size_t offset) const {
    return st(this).slots.at(blockId)->readFromBlock(offset);
}

void GPU::writeToBlock(std::size_t blockId, std::size_t offset, int value) {
    st(this).slots.at(blockId)->writeToBlock(offset, value);
}

PrefixMatch GPU::findLongestCachedPrefix(const std::string& modelId, std::size_t embeddingSize,
                                         const std::vector<std::string>& promptTokens) const {
    const FakeState& s = states.at(this);
    PrefixMatch m = {0, std::vector<std::size_t>()};
    std::size_t t = blockSize / embeddingSize;
    for (std::size_t len = t; len <= promptTokens.size(); len += t) {
        std::vector<std::string> key(promptTokens.begin(), promptTokens.begin() + len);
        bool found = false;
        for (std::size_t i = 0; i < s.slots.size() && !found; i++) {
            if (s.slots[i] != nullptr && s.hasKey[i] && s.keyModel[i] == modelId && s.keyTokens[i] == key) {
                m.blockIds.push_back(i);
                m.matchedTokens = len;
                found = true;
            }
        }
        if (!found) break;
    }
    return m;
}

// ---------- פונקציות עזר לבדיקות ----------
int fakeRefCount(const GPU& g, std::size_t slot) { return states.at(&g).refs[slot]; }
bool fakeOccupied(const GPU& g, std::size_t slot) { return states.at(&g).slots[slot] != nullptr; }
bool fakeHasKey(const GPU& g, std::size_t slot) { return states.at(&g).hasKey[slot]; }
std::vector<std::string> fakeKey(const GPU& g, std::size_t slot) { return states.at(&g).keyTokens[slot]; }
long fakeReserved(const GPU& g, int requestId) {
    const FakeState& s = states.at(&g);
    std::map<int, long>::const_iterator it = s.reservedById.find(requestId);
    return it == s.reservedById.end() ? 0 : it->second;
}
long fakeTotalReserved(const GPU& g) { return states.at(&g).totalReserved; }
int fakeViolations(const GPU& g) { return states.at(&g).violations; }
std::size_t fakeBlockCount(const GPU& g) {
    const FakeState& s = states.at(&g);
    std::size_t c = 0;
    for (std::size_t i = 0; i < s.slots.size(); i++) if (s.slots[i] != nullptr) c++;
    return c;
}
void fakeCleanup(const GPU& g) {   // GPU.h עוד בלי destructor: מוחקים כאן כדי ש-valgrind יהיה נקי
    FakeState& s = states.at(&g);
    for (std::size_t i = 0; i < s.slots.size(); i++) delete s.slots[i];
    states.erase(&g);
}
