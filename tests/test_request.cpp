// ============================================================================
// בדיקות ל-Request, GenerativeRequest ו-EmbeddingRequest.
// לא חלק מההגשה! רץ מול GPU מזויף (tests/fake_gpu.cpp).
//
// קומפילציה (מהתיקייה spl-data-center):
//   g++ -Wall -Weffc++ -std=c++11 -g -Iinclude tests/test_request.cpp tests/fake_gpu.cpp src/Request.cpp
//       src/Model.cpp src/MemoryBlock.cpp -o bin/test_request
//   ./bin/test_request
//   valgrind --leak-check=full --show-reachable=yes ./bin/test_request
// ============================================================================
#include "Request.h"
#include "Model.h"
#include "GPU.h"
#include "MemoryBlock.h"
#include <iostream>
#include <string>
#include <vector>

// פונקציות עזר מה-GPU המזויף
int fakeRefCount(const GPU& g, std::size_t slot);
bool fakeOccupied(const GPU& g, std::size_t slot);
bool fakeHasKey(const GPU& g, std::size_t slot);
std::vector<std::string> fakeKey(const GPU& g, std::size_t slot);
long fakeReserved(const GPU& g, int requestId);
long fakeTotalReserved(const GPU& g);
int fakeViolations(const GPU& g);
std::size_t fakeBlockCount(const GPU& g);
void fakeCleanup(const GPU& g);

static int failures = 0;
static void check(bool cond, const std::string& name) {
    std::cout << (cond ? "  [PASS] " : "  [FAIL] ") << name << std::endl;
    if (!cond) failures++;
}
static void section(const std::string& name) { std::cout << "\n== " << name << " ==" << std::endl; }

static std::vector<std::string> vec(const std::string& s) {   // "a b c" -> {a,b,c}
    std::vector<std::string> v; std::string cur;
    for (std::size_t i = 0; i < s.size(); i++) {
        if (s[i] == ' ') { if (!cur.empty()) { v.push_back(cur); cur.clear(); } }
        else cur += s[i];
    }
    if (!cur.empty()) v.push_back(cur);
    return v;
}

// הווקטור ש-generateTokenEmbedding מייצרת לרצף, מחושב בנפרד על GPU זמני
static std::vector<int> embeddingOf(const Model& m, const std::vector<std::string>& seq, std::size_t e) {
    GPU scratch(1, e);
    EmbeddingRequest dummy(-1, m, 0, "x");
    scratch.reserve(1, dummy);
    std::size_t slot = scratch.allocateBlock(dummy);
    m.generateTokenEmbedding(seq, scratch, slot, 0);
    std::vector<int> v;
    for (std::size_t i = 0; i < e; i++) v.push_back(scratch.readFromBlock(slot, i));
    fakeCleanup(scratch);
    return v;
}

static void runToEnd(Request& r, GPU& g) {
    int guard = 0;
    while (!r.finished() && guard < 1000) { r.act(g); guard++; }
}

int main() {
    // ------------------------------------------------------------------
    section("1. Constructors, getters, tokenizer");
    {
        Model m("llama", 2);
        GenerativeRequest g(7, m, 3, "Hello, World! a-b", "x y z", 2);
        check(g.getId() == 7, "getId");
        check(g.getAdmissionCycle() == 3, "getAdmissionCycle");
        check(&g.getModel() == &m, "getModel returns the same model object");
        check(g.getCacheableTokens() == vec("Hello World ab"), "prompt tokenized: punctuation removed, split by spaces");
        check(g.getOutput() == "x y", "scripted output cut to maxOutputTokens (3 -> 2)");
        check(!g.finished(), "new generative request is not finished");

        GenerativeRequest g2(8, m, 0, "a b", "x", 5);
        check(g2.getOutput() == "x", "scripted output shorter than max is kept whole");

        EmbeddingRequest e(9, m, 1, "hi there");
        check(e.getCacheableTokens().empty(), "embedding: getCacheableTokens is empty");
        check(!e.finished(), "new embedding request is not finished");
        check(e.getOutput() == "[]", "embedding output before running is []");
    }

    // ------------------------------------------------------------------
    section("2. Memory calculations (no cache)");
    {
        Model m("llama", 2);
        GenerativeRequest g(1, m, 0, "a b c", "x y", 4);
        check(g.getMaxNeededMemory() == 2 * (3 + 4), "generative getMaxNeededMemory = e*(n+m), m = maxOutputTokens");
        EmbeddingRequest e(2, m, 0, "one two three four five");
        check(e.getMaxNeededMemory() == 4, "embedding getMaxNeededMemory = 2*e, regardless of prompt length");
        GPU gpu(10, 4);
        check(g.getNeededMemory(gpu) == 14, "generative getNeededMemory on empty GPU = max");
        check(e.getNeededMemory(gpu) == 4, "embedding getNeededMemory = 2*e");
        fakeCleanup(gpu);
    }

    // ------------------------------------------------------------------
    section("3. Embedding: full run (e=2, B=4 -> one workspace block)");
    {
        Model m("e5", 2);
        GPU gpu(10, 4);
        EmbeddingRequest r(1, m, 0, "hi there friend");
        r.reserveMemory(gpu);
        check(fakeBlockCount(gpu) == 1, "reserveMemory allocates 1 workspace block");
        check(fakeRefCount(gpu, 0) == 1, "workspace is held once (refCount 1)");
        check(r.getNeedToUnreserve() == 0, "nothing left to unreserve after allocation");

        int acts = 0;
        while (!r.finished()) { r.act(gpu); acts++; }
        check(acts == 3, "one act per prompt token (3 acts)");

        std::vector<int> a = embeddingOf(m, vec("hi"), 2), b = embeddingOf(m, vec("there"), 2),
                         c = embeddingOf(m, vec("friend"), 2);
        std::string expected = "[" + std::to_string((a[0] + b[0] + c[0]) / 3) + ", " +
                               std::to_string((a[1] + b[1] + c[1]) / 3) + "]";
        check(r.getOutput() == expected, "output = mean of single-token vectors: " + expected);

        r.act(gpu);
        check(r.getOutput() == expected, "extra act() after finishing changes nothing");
#ifndef NO_DEALLOC
        r.deallocateMemory(gpu);
        check(fakeBlockCount(gpu) == 0, "after deallocateMemory the workspace is destroyed");
        check(fakeTotalReserved(gpu) == 0, "no reservation left on the GPU");
#endif
        check(fakeViolations(gpu) == 0, "no invalid GPU usage");
        fakeCleanup(gpu);
    }

    // ------------------------------------------------------------------
    section("4. Embedding: e == B -> two workspace blocks");
    {
        Model m("e5", 4);
        GPU gpu(10, 4);
        EmbeddingRequest r(1, m, 0, "alpha beta");
        r.reserveMemory(gpu);
        check(fakeBlockCount(gpu) == 2, "reserveMemory allocates 2 blocks when 2e > B");
        runToEnd(r, gpu);
        std::vector<int> a = embeddingOf(m, vec("alpha"), 4), b = embeddingOf(m, vec("beta"), 4);
        std::string expected = "[";
        for (int j = 0; j < 4; j++) { expected += std::to_string((a[j] + b[j]) / 2); if (j < 3) expected += ", "; }
        expected += "]";
        check(r.getOutput() == expected, "mean is correct across two blocks: " + expected);
#ifndef NO_DEALLOC
        r.deallocateMemory(gpu);
        check(fakeBlockCount(gpu) == 0, "both workspace blocks destroyed");
#endif
        check(fakeViolations(gpu) == 0, "no invalid GPU usage");
        fakeCleanup(gpu);
    }

    // ------------------------------------------------------------------
    section("5. Embedding: empty prompt after tokenizer");
    {
        Model m("e5", 2);
        EmbeddingRequest r(1, m, 0, "!!! ...");
        check(r.finished(), "prompt with only punctuation -> finished immediately");
        check(r.getOutput() == "[]", "output is [] (and no division by zero)");
    }

    // ------------------------------------------------------------------
    section("6. Generative: full run, no cache (t=2, prompt 'a b c', output 'x y')");
    {
        Model m("llama", 2);
        GPU gpu(10, 4);
        GenerativeRequest r(1, m, 0, "a b c", "x y", 2);
        r.reserveMemory(gpu);
        check(fakeReserved(gpu, 1) == 3, "reserves ceil(2*(3+2)/4) = 3 blocks");
        check(fakeBlockCount(gpu) == 0, "generative does not allocate in reserveMemory");

        int acts = 0;
        while (!r.finished()) { r.act(gpu); acts++; }
        check(acts == 5, "5 acts (3 prompt + 2 output)");
        check(r.getOutput() == "x y", "output 'x y'");
        check(fakeBlockCount(gpu) == 3, "3 blocks were created");
        check(fakeHasKey(gpu, 0) && fakeKey(gpu, 0) == vec("a b"), "block 0 key = 'a b'");
        check(fakeHasKey(gpu, 1) && fakeKey(gpu, 1) == vec("a b c x"), "block 1 key = 'a b c x'");
        check(!fakeHasKey(gpu, 2), "last block [y] is partial -> no key");

        std::vector<int> exp = embeddingOf(m, vec("a b c x"), 2);
        check(gpu.readFromBlock(1, 2) == exp[0] && gpu.readFromBlock(1, 3) == exp[1],
              "KV vector of 'x' depends on the whole prefix and sits at block 1, offset 2");
        std::vector<int> a1 = embeddingOf(m, vec("a"), 2);
        check(gpu.readFromBlock(0, 0) == a1[0] && gpu.readFromBlock(0, 1) == a1[1], "KV vector of 'a' at block 0, offset 0");
        check(r.getNeedToUnreserve() == 0, "all 3 reserved blocks were used");
#ifndef NO_DEALLOC
        r.deallocateMemory(gpu);
        check(fakeOccupied(gpu, 0) && fakeRefCount(gpu, 0) == 0, "block 'a b' stays as idle (cached)");
        check(fakeOccupied(gpu, 1) && fakeRefCount(gpu, 1) == 0, "block 'a b c x' stays as idle (cached)");
        check(!fakeOccupied(gpu, 2), "partial block without key is deleted");
        check(fakeTotalReserved(gpu) == 0, "no reservation left");
#endif
        check(fakeViolations(gpu) == 0, "no invalid GPU usage");
        fakeCleanup(gpu);
    }

    // ------------------------------------------------------------------
    section("7. Generative: prefix cache (the example from section 4.2)");
#ifndef NO_DEALLOC
    {
        Model m("llama", 2);
        GPU gpu(10, 4);
        GenerativeRequest r1(1, m, 0, "a b c", "", 0);
        r1.reserveMemory(gpu); runToEnd(r1, gpu); r1.deallocateMemory(gpu);
        check(fakeBlockCount(gpu) == 1 && fakeKey(gpu, 0) == vec("a b"), "request 1 leaves only cached 'a b'");

        GenerativeRequest r2(2, m, 0, "a b c", "x y", 2);
        check(r2.getNeededMemory(gpu) == 2 * (3 - 2 + 2), "request 2: getNeededMemory uses h = 2");
        r2.reserveMemory(gpu);
        check(fakeRefCount(gpu, 0) == 1, "request 2 acquired the cached block 'a b'");
        check(fakeReserved(gpu, 2) == 2, "request 2 reserves only 2 new blocks");
        int acts = 0;
        while (!r2.finished()) { r2.act(gpu); acts++; }
        check(acts == 3, "request 2 computes only c, x, y (3 acts)");
        r2.deallocateMemory(gpu);
        check(fakeRefCount(gpu, 0) == 0 && fakeOccupied(gpu, 0), "'a b' idle again after request 2");

        GenerativeRequest r3(3, m, 0, "a b c x y z", "", 0);
        check(r3.getNeededMemory(gpu) == 2 * (6 - 4 + 0), "request 3: cached prefix of 4 tokens (a b + a b c x)");
        r3.reserveMemory(gpu);
        acts = 0;
        while (!r3.finished()) { r3.act(gpu); acts++; }
        check(acts == 2, "request 3 computes only y, z (2 acts)");
        r3.deallocateMemory(gpu);
        check(fakeViolations(gpu) == 0, "no invalid GPU usage");
        fakeCleanup(gpu);
    }
#else
    std::cout << "  (skipped: needs deallocateMemory)" << std::endl;
#endif

    // ------------------------------------------------------------------
    section("8. Generative: output shorter than maxOutputTokens");
    {
        Model m("llama", 2);
        GPU gpu(10, 4);
        GenerativeRequest r(1, m, 0, "a b", "x", 6);
        r.reserveMemory(gpu);
        check(fakeReserved(gpu, 1) == 4, "reserves for the worst case: ceil(2*(2+6)/4) = 4");
        runToEnd(r, gpu);
        check(r.getOutput() == "x", "output 'x'");
        check(r.getNeedToUnreserve() == 2, "used 2 of 4 reserved blocks -> 2 to unreserve");
#ifndef NO_DEALLOC
        r.deallocateMemory(gpu);
        check(fakeReserved(gpu, 1) == 0 && fakeTotalReserved(gpu) == 0, "deallocateMemory returns the unused reservation");
#endif
        check(fakeViolations(gpu) == 0, "no invalid GPU usage");
        fakeCleanup(gpu);
    }

    // ------------------------------------------------------------------
    section("9. Two requests computing the same prefix at the same time");
#ifndef NO_DEALLOC
    {
        Model m("llama", 2);
        GPU gpu(10, 4);
        GenerativeRequest r1(1, m, 0, "a b", "", 0), r2(2, m, 0, "a b", "", 0);
        r1.reserveMemory(gpu); r2.reserveMemory(gpu);
        runToEnd(r1, gpu); runToEnd(r2, gpu);
        check(fakeHasKey(gpu, 0) && !fakeHasKey(gpu, 1), "only the first block gets key 'a b'; the duplicate stays without key");
        r1.deallocateMemory(gpu); r2.deallocateMemory(gpu);
        check(fakeOccupied(gpu, 0) && !fakeOccupied(gpu, 1), "duplicate block is deleted, original stays cached");
        fakeCleanup(gpu);
    }
#else
    std::cout << "  (skipped: needs deallocateMemory)" << std::endl;
#endif

    // ------------------------------------------------------------------
    section("10. clone(const Model&) - also in the middle of a run");
    {
        Model m1("llama", 2), m2("llama", 2);
        GPU gpuA(10, 4), gpuB(10, 4);
        GenerativeRequest r(1, m1, 5, "a b c", "x y", 2);
        Request* c0 = r.clone(m2);
        check(&c0->getModel() == &m2, "clone refers to the new model");
        check(c0->getId() == 1 && c0->getAdmissionCycle() == 5, "clone keeps id and admissionCycle");
        check(c0->getOutput() == "x y" && c0->getCacheableTokens() == r.getCacheableTokens(), "clone keeps tokens and output");
        check(dynamic_cast<GenerativeRequest*>(c0) != nullptr, "clone of GenerativeRequest is a GenerativeRequest");
        delete c0;   // through Request* -> needs virtual destructor (valgrind checks)

        // mid-run: two identical requests on two GPUs, clone one after 2 acts, finish both
        GenerativeRequest a(1, m1, 0, "a b c", "x y", 2), b(1, m1, 0, "a b c", "x y", 2);
        a.reserveMemory(gpuA); b.reserveMemory(gpuB);
        a.act(gpuA); a.act(gpuA); b.act(gpuB); b.act(gpuB);
        Request* mid = b.clone(m2);
        check(!mid->finished(), "mid-run clone is not finished");
        runToEnd(a, gpuA);
        // the clone continues on gpuB (same slots), b itself is abandoned
        runToEnd(*mid, gpuB);
        check(mid->getOutput() == a.getOutput(), "mid-run clone finishes with the same output");
        check(gpuB.readFromBlock(1, 2) == gpuA.readFromBlock(1, 2), "mid-run clone wrote the same KV vectors");
        delete mid;

        EmbeddingRequest e(2, m1, 0, "hi there");
        Request* ce = e.clone(m2);
        check(dynamic_cast<EmbeddingRequest*>(ce) != nullptr && &ce->getModel() == &m2, "clone of EmbeddingRequest keeps type, new model");
        delete ce;
        fakeCleanup(gpuA); fakeCleanup(gpuB);
    }

    // ------------------------------------------------------------------
    std::cout << "\n" << (failures == 0 ? "ALL PASSED" : std::to_string(failures) + " FAILED") << std::endl;
    return failures;
}
