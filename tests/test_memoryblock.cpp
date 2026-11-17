#include "MemoryBlock.h"
#include <iostream>
#include <utility>
#include <string>

static int failures = 0;
static void check(bool cond, const std::string& name) {
    std::cout << (cond ? "[PASS] " : "[FAIL] ") << name << std::endl;
    if (!cond) failures++;
}

int main() {
    // 1. בנאי: כל הערכים 0
    {
        KVBlock b(4);
        bool allZero = true;
        for (size_t i = 0; i < 4; i++) if (b.readFromBlock(i) != 0) allZero = false;
        check(allZero, "constructor: all values are 0");
    }
    // 2. קריאה וכתיבה
    {
        KVBlock b(4);
        b.writeToBlock(2, 42);
        check(b.readFromBlock(2) == 42, "write/read");
    }
    // 3. copy constructor: נתונים נפרדים
    {
        KVBlock a(4);
        a.writeToBlock(0, 10);
        KVBlock b(a);
        b.writeToBlock(0, 99);
        check(a.readFromBlock(0) == 10 && b.readFromBlock(0) == 99, "copy ctor: independent data");
    }
    // 4. copy constructor: refCount של העותק הוא 0
    //    (המקור מוחזק פעמיים; אם העותק מתחיל ב-0, acquire+release אחד מביא אותו ל-0 -> Destroy)
    {
        EmbeddingWorkspace a(4);
        a.acquire(); a.acquire();
        EmbeddingWorkspace b(a);
        b.acquire();
        check(b.release() == IdleAction::Destroy, "copy ctor: copy starts with refCount 0");
    }
    // 5. copy assignment
    {
        KVBlock a(4), b(8);
        a.writeToBlock(1, 5);
        b = a;
        b.writeToBlock(1, 6);
        check(a.readFromBlock(1) == 5 && b.readFromBlock(1) == 6, "copy assignment: independent data");
        b = b; // self assignment
        check(b.readFromBlock(1) == 6, "copy assignment: self assignment safe");
    }
    // 6. move constructor
    {
        KVBlock a(4);
        a.writeToBlock(3, 7);
        KVBlock b(std::move(a));
        check(b.readFromBlock(3) == 7, "move ctor: data transferred");
    } // a (moved-from) destroyed here -> valgrind checks no double free
    // 7. move assignment
    {
        KVBlock a(4), b(4);
        a.writeToBlock(0, 3);
        b = std::move(a);
        check(b.readFromBlock(0) == 3, "move assignment: data transferred");
        KVBlock& same = b;
        b = std::move(same);              // השמה עצמית ב-move
        check(b.readFromBlock(0) == 3, "move assignment: self move safe");
    }
    // 8. clone שומר טיפוס ונתונים
    {
        MemoryBlock* k = new KVBlock(4);
        MemoryBlock* e = new EmbeddingWorkspace(4);
        k->writeToBlock(0, 11);
        MemoryBlock* k2 = k->clone();
        MemoryBlock* e2 = e->clone();
        k2->writeToBlock(0, 12);
        check(k->readFromBlock(0) == 11 && k2->readFromBlock(0) == 12, "clone: independent data");
        k2->acquire(); e2->acquire();
        check(k2->release() == IdleAction::KeepCached, "clone: KVBlock stays KVBlock");
        check(e2->release() == IdleAction::Destroy, "clone: EmbeddingWorkspace stays EmbeddingWorkspace");
        delete k; delete e; delete k2; delete e2;   // delete through base pointer
    }
    // 9. acquire / release
    {
        EmbeddingWorkspace w(4);
        w.acquire(); w.acquire();
        IdleAction first = w.release();   // still held
        IdleAction second = w.release();  // last
        check(first == IdleAction::KeepCached, "release: still held -> KeepCached");
        check(second == IdleAction::Destroy, "release: last release -> onLastRelease()");
    }
    // 10. גישה מחוץ לטווח זורקת חריגה
    {
        KVBlock b(4);
        bool threw = false;
        try { b.readFromBlock(4); } catch (const std::exception&) { threw = true; }
        check(threw, "readFromBlock out of range throws");
    }

    std::cout << (failures == 0 ? "ALL PASSED" : "SOME FAILED") << std::endl;
    return failures;
}
