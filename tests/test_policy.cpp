// ============================================================================
// בדיקות ל-SchedulingPolicy (ArrivalOrder, ShortestJob, LongestPrefix).
// לא חלק מההגשה! רץ מול GPU מזויף (tests/fake_gpu.cpp).
//
// קומפילציה (מהתיקייה spl-data-center):
//   g++ -Wall -Weffc++ -std=c++11 -g -Iinclude tests/test_policy.cpp tests/fake_gpu.cpp src/SchedulingPolicy.cpp
//       src/Request.cpp src/Model.cpp src/MemoryBlock.cpp -o bin/test_policy
//   ./bin/test_policy
//   valgrind --leak-check=full --show-reachable=yes ./bin/test_policy
// ============================================================================
#include "SchedulingPolicy.h"
#include "Request.h"
#include "Model.h"
#include "GPU.h"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

void fakeCleanup(const GPU& g);

static int failures = 0;
static void check(bool cond, const std::string& name) {
    std::cout << "  " << (cond ? "[PASS] " : "[FAIL] ") << name << std::endl;
    if (!cond) failures++;
}
static void section(const std::string& name) { std::cout << "\n== " << name << " ==" << std::endl; }

// מתרגם את האינדקסים ש-choose מחזירה ל-ids, כדי שיהיה קל להשוות לדוגמה ב-PDF
static std::vector<int> ids(const std::vector<int>& order, const std::vector<const Request*>& eligible) {
    std::vector<int> v;
    for (std::size_t i = 0; i < order.size(); i++) v.push_back(eligible[order[i]]->getId());
    return v;
}
static std::string str(const std::vector<int>& v) {
    std::string s;
    for (std::size_t i = 0; i < v.size(); i++) { if (i) s += ","; s += std::to_string(v[i]); }
    return s;
}
static bool isPermutation(const std::vector<int>& order, std::size_t k) {
    if (order.size() != k) return false;
    std::vector<int> sorted(order);
    std::sort(sorted.begin(), sorted.end());
    for (std::size_t i = 0; i < k; i++) if (sorted[i] != static_cast<int>(i)) return false;
    return true;
}
// ממלא את ה-cache של gpu ב-prefix keys של prompt (מריץ בקשה בלי פלט ומשחרר אותה)
static void seedCache(GPU& gpu, const Model& m, const std::string& prompt) {
    GenerativeRequest seed(1000, m, 0, prompt, "", 0);
    seed.reserveMemory(gpu);
    while (!seed.finished()) seed.act(gpu);
    seed.deallocateMemory(gpu);
}
static void expectOrder(SchedulingPolicy* p, const std::vector<const Request*>& eligible,
                        const std::vector<GPU>& gpus, const std::string& expected, const std::string& name) {
    std::vector<int> order = p->choose(eligible, gpus);
    std::string got = str(ids(order, eligible));
    check(isPermutation(order, eligible.size()) && got == expected,
          name + ": expected " + expected + ", got " + got);
}

int main() {
    // ------------------------------------------------------------------
    section("1. The example from the PDF (section 4.4.6)");
    {
        // בלוק אחד = טוקן אחד: embeddingSize == blockSize
        Model llama("llama", 2), e5("e5", 2);
        std::vector<GPU> gpus;
        gpus.reserve(1);
        gpus.emplace_back(20, 2);
        seedCache(gpus[0], llama, "the cat sat");   // keys: "the", "the cat", "the cat sat"

        GenerativeRequest r3(3, llama, 0, "a dog", "o o o", 3);
        EmbeddingRequest  r4(4, e5, 0, "hi");
        GenerativeRequest r5(5, llama, 0, "the cat ran", "o", 1);
        GenerativeRequest r6(6, llama, 0, "the cat sat down", "o o o o", 4);
        std::vector<const Request*> eligible;
        eligible.push_back(&r3); eligible.push_back(&r4); eligible.push_back(&r5); eligible.push_back(&r6);

        SchedulingPolicy* arrival  = new ArrivalOrderPolicy();
        SchedulingPolicy* shortest = new ShortestJobPolicy();
        SchedulingPolicy* longest  = new LongestPrefixPolicy();
        expectOrder(arrival,  eligible, gpus, "3,4,5,6", "arrivalOrder");
        expectOrder(shortest, eligible, gpus, "4,5,3,6", "shortestJob");
        expectOrder(longest,  eligible, gpus, "6,5,3,4", "longestPrefix (3 and 4 tie with h=0 -> lower id)");

        // אותן בקשות בסדר קלט אחר: התוצאה (לפי ids) חייבת להיות זהה
        std::vector<const Request*> shuffled;
        shuffled.push_back(&r6); shuffled.push_back(&r4); shuffled.push_back(&r3); shuffled.push_back(&r5);
        expectOrder(arrival,  shuffled, gpus, "3,4,5,6", "arrivalOrder, shuffled input");
        expectOrder(shortest, shuffled, gpus, "4,5,3,6", "shortestJob, shuffled input");
        expectOrder(longest,  shuffled, gpus, "6,5,3,4", "longestPrefix, shuffled input");

        delete arrival; delete shortest; delete longest;
        fakeCleanup(gpus[0]);
    }

    // ------------------------------------------------------------------
    section("2. ArrivalOrder: lower admissionCycle first, ties by id");
    {
        Model m("llama", 2);
        std::vector<GPU> gpus; gpus.reserve(1); gpus.emplace_back(10, 4);
        GenerativeRequest a(1, m, 5, "x", "", 0), b(9, m, 2, "x", "", 0), c(3, m, 2, "x", "", 0), d(2, m, 7, "x", "", 0);
        std::vector<const Request*> eligible;
        eligible.push_back(&a); eligible.push_back(&b); eligible.push_back(&c); eligible.push_back(&d);
        SchedulingPolicy* p = new ArrivalOrderPolicy();
        expectOrder(p, eligible, gpus, "3,9,1,2", "cycles 5,2,2,7 -> ids 3,9 (cycle 2), 1 (cycle 5), 2 (cycle 7)");
        delete p;
        fakeCleanup(gpus[0]);
    }

    // ------------------------------------------------------------------
    section("3. ShortestJob compares BLOCKS, not raw memory");
    {
        Model m("llama", 1);                          // e = 1
        std::vector<GPU> gpus; gpus.reserve(1); gpus.emplace_back(10, 4);   // B = 4
        GenerativeRequest a(2, m, 0, "a b c", "x y", 2);         // 1*(3+2) = 5 ints -> 2 blocks
        GenerativeRequest b(1, m, 0, "a b c d e f", "x y", 2);   // 1*(6+2) = 8 ints -> 2 blocks
        GenerativeRequest c(3, m, 0, "a", "", 0);                // 1 int -> 1 block
        std::vector<const Request*> eligible;
        eligible.push_back(&a); eligible.push_back(&b); eligible.push_back(&c);
        SchedulingPolicy* p = new ShortestJobPolicy();
        expectOrder(p, eligible, gpus, "3,1,2", "a (5 ints) and b (8 ints) both need 2 blocks -> tie -> lower id first");
        delete p;
        fakeCleanup(gpus[0]);
    }

    // ------------------------------------------------------------------
    section("4. ShortestJob uses maxOutputTokens, ignores the cache");
    {
        Model m("llama", 2);
        std::vector<GPU> gpus; gpus.reserve(1); gpus.emplace_back(20, 2);   // t = 1
        seedCache(gpus[0], m, "p q r s");
        GenerativeRequest cached(1, m, 0, "p q r s", "", 0);   // max 4 blocks (needs 0 new)
        GenerativeRequest fresh(2, m, 0, "z", "o", 2);         // max 3 blocks
        std::vector<const Request*> eligible;
        eligible.push_back(&cached); eligible.push_back(&fresh);
        SchedulingPolicy* p = new ShortestJobPolicy();
        expectOrder(p, eligible, gpus, "2,1", "fully cached request is still 'longer' (4 > 3 blocks)");
        delete p;
        fakeCleanup(gpus[0]);
    }

    // ------------------------------------------------------------------
    section("5. LongestPrefix: maximum over all GPUs, same model only, embedding h=0");
    {
        Model llama("llama", 2), other("other", 2), e5("e5", 2);
        std::vector<GPU> gpus; gpus.reserve(2);
        gpus.emplace_back(20, 4);    // t = 2
        gpus.emplace_back(20, 4);
        seedCache(gpus[0], llama, "a b");          // gpu 0: "a b"
        seedCache(gpus[1], llama, "a b c d");      // gpu 1: "a b", "a b c d"
        GenerativeRequest r1(1, llama, 0, "a b c d e", "", 0);    // h = 4 (on gpu 1)
        GenerativeRequest r2(2, llama, 0, "a b x", "", 0);        // h = 2
        GenerativeRequest r3(3, other, 0, "a b c d e", "", 0);    // other model -> h = 0
        EmbeddingRequest  r4(4, e5, 0, "a b c d");                // embedding -> h = 0
        std::vector<const Request*> eligible;
        eligible.push_back(&r3); eligible.push_back(&r4); eligible.push_back(&r2); eligible.push_back(&r1);
        SchedulingPolicy* p = new LongestPrefixPolicy();
        expectOrder(p, eligible, gpus, "1,2,3,4", "h = 4 (gpu 1), 2, then 0,0 by id");
        delete p;
        fakeCleanup(gpus[0]); fakeCleanup(gpus[1]);
    }

    // ------------------------------------------------------------------
    section("6. LongestPrefix: only full blocks count");
    {
        Model m("llama", 2);
        std::vector<GPU> gpus; gpus.reserve(1); gpus.emplace_back(20, 4);   // t = 2
        seedCache(gpus[0], m, "a b c d");   // keys "a b", "a b c d"
        GenerativeRequest r1(1, m, 0, "a b c", "", 0);    // "a b" matches, "a b c" is not a full block -> h = 2
        GenerativeRequest r2(2, m, 0, "a b c d", "", 0);  // h = 4
        std::vector<const Request*> eligible;
        eligible.push_back(&r1); eligible.push_back(&r2);
        SchedulingPolicy* p = new LongestPrefixPolicy();
        expectOrder(p, eligible, gpus, "2,1", "h = 4 before h = 2");
        delete p;
        fakeCleanup(gpus[0]);
    }

    // ------------------------------------------------------------------
    section("7. clone() and empty input");
    {
        Model m("llama", 2);
        std::vector<GPU> gpus; gpus.reserve(1); gpus.emplace_back(10, 4);
        SchedulingPolicy* ps[3] = { new ArrivalOrderPolicy(), new ShortestJobPolicy(), new LongestPrefixPolicy() };
        for (int i = 0; i < 3; i++) {
            SchedulingPolicy* c = ps[i]->clone();
            bool sameType = (i == 0 && dynamic_cast<ArrivalOrderPolicy*>(c) != nullptr) ||
                            (i == 1 && dynamic_cast<ShortestJobPolicy*>(c) != nullptr) ||
                            (i == 2 && dynamic_cast<LongestPrefixPolicy*>(c) != nullptr);
            check(sameType && c != ps[i], "clone #" + std::to_string(i) + " is a new object of the same type");
            std::vector<const Request*> none;
            check(c->choose(none, gpus).empty(), "choose on empty list returns empty (policy #" + std::to_string(i) + ")");
            delete c;          // through SchedulingPolicy* -> virtual destructor
            delete ps[i];
        }
        fakeCleanup(gpus[0]);
    }

    std::cout << "\n" << (failures == 0 ? "ALL PASSED" : std::to_string(failures) + " FAILED") << std::endl;
    return failures;
}
