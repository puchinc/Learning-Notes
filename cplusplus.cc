#include <iostream>
/////////////////////
//  SMART POINTERS
/////////////////////

#include <memory>

// Linked List
struct ListNode {
  int data;
  std::shared_ptr<ListNode> next;

  ~ListNode() {
    while (pop())
      ;
  }
  ListNode(int n) { data = n; }
  void insert(int n) {
    if (next)
      next->insert(n);
    else
      next.reset(new ListNode(n));
  }
  std::shared_ptr<ListNode> pop() {
    if (!next) return nullptr;
    return next = next->next;
  }
};

// Comparison functions
// We also need a function to compare two such objects. Actually, we need two -
// one for a container that holds object by value, and another for the
// by-pointer version:
struct Data {
  int id = 0;
};

bool compare_by_value(const Data& a, const Data& b) { return a.id < b.id; }

bool compare_by_ptr(const Data* a, const Data* b) { return a->id < b->id; }

bool compare_by_uniqptr(const std::unique_ptr<Data>& a,
                        const std::unique_ptr<Data>& b) {
  return a->id < b->id;
}

/////////////////////
// LAMBDA
// /////////////////

#if USING_LAMBDA
const bool registered = [this, &k]() -> bool {
  for (const auto& reg : registries_)
    for (const auto& elem : reg)
      if (elem == k) return true;
  return false;
}();
#else   // !USING_LAMBDA
bool registered = false;  // can't be const
for (const auto& reg : registries_) {
  if (registered) break;  // error-prone
  for (const auto& elem : reg) {
    if (elem == k) {
      registered = true;
      break;  // C++ only has single-level break
    }
  }
}
#endif  // !USING_LAMBDA

/////////////////////
//  DATA STRUCTURE
/////////////////////
#include <string>

// Namespace
namespace space {
int a = 10;
}

// can use all members in space
#include <iostream>
using namespace std;

// only use specific memeber
using space::a;

// aliase
namespace s = space;

// static global variable
namespace {
int count = 0;
}

// String
//
// Raw string literal
string s1 = "Need escape \n";
string s2 = R"(No needs escape
)";

// File path operation
std::string path = "";
std::string base_filename = path.substr(path.find_last_of("/") + 1);
size_t lastindex = base_filename.find_last_of(".");
string rawname = base_filename.substr(0, lastindex);

// STL
// Container
#include <vector>
std::vector<std::string> strings;
strings.push_back("abc");
// vec.front();
// vec.back();
// vec.emtpy();

#include <unordered_map>
std::unordered_map<std::string, int> m;

auto search = m.find("element");
if (search != m.end()) {
  cout << "find element" << endl;
} else {
  string key = search->first;
  int value = search->second;
}

////////////////////
//  CONTROL FLOW
/////////////////////

for (int i = 0; i < v.size(); ++i) {
  cout << v[i] << endl;
}

for (const auto& it : container) {
  cout << it << endl;
}

for (vector<string>::iterator it = v.begin(); it != v.end(); ++it) {
  // for (auto it = v.begin(); it != v.end(); ++it) {
  cout << *it << endl;
}

////////////////////
//  Optimization
/////////////////////

// https://medium.com/@samathy_barratt/constantly-confusing-c-const-and-constexpr-pointer-behaviour-7eddb041e15a
// constexpr, evaluate once in compile time,
//    reducing runtime

// lvalue, exists address to access
int lvalue;

// rvalue, only bind to value
int dummy = lvalue + 3;  // lvalue + 3 is rvalue

// left value reference
int& lr = lvalue;
const int& lr = l_or_r;
void foo(const string& str);  // accept lvalue

// right value reference
int&& rr = 10;
void foo(string&& str);  // accept rvalue, be triggered

// universal reference, ur can be l_or_r
template <typename T>
void f(T&& ur);

auto&& ur = l_or_r;

// reference collapsing
T& T& = T&;
T&& T& = T&;
T& T&& = T&;
T&& T&& = T&&;

// https://zhuanlan.zhihu.com/p/54050093
// Move semantic std::move(T l_or_r)
//    make l_or_r rvalue
int lvalue = 10;
int l = std::move(lvalue);
// now lvalue become rvalue

// Forward semantic std::forward(T l_or_r)
//    l still lvalue, r still rvalue
template <typename T>
void wrapper(T&& param) {
  foo(std::forward<T>(param));  // perfect forwarding
}

///////////////////////
//      CLASS
////////////////////////
// Inheritance:

// This class inherits everything public and protected from the Dog class
// as well as private but may not directly access private members/methods
// without a public or protected method for doing so
class OwnedDog : public Dog {
 public:
  void setOwner(const std::string& dogsOwner);
  // The override keyword is optional but makes sure you are actually
  // overriding the method in a base class.
  void print() const override;

 private:
  std::string owner;
};

// Meanwhile, in the corresponding .cpp file:

// Objects (such as strings) should be passed by reference
// if you are modifying them or const reference if you are not.
void OwnedDog::setOwner(const std::string& dogsOwner) { owner = dogsOwner; }

void OwnedDog::print() const {
  Dog::print();  // Call the print function in the base Dog class
  std::cout << "Dog is owned by " << owner << "\n";
  // Prints "Dog is <name> and weights <weight>"
  //        "Dog is owned by <owner>"
}

// Constructor

class X {
  X() = default;  // new instance default constructor
  X(int num);
  void func() = delete;  // cannot use func()

  X(const X&) : length(rhs.length);  // copy constructor
  X(const X&) = delete;              // disallow copy

  X& operator=(const X&);           // copy assignment constructor
  X& operator=(const X&) = delete;  // disallow copy assignment

  X(X&&);           // move constructor
  X(X&&) = delete;  // disallow move

  X& operator=(const X&&);           // move assignment constructor
  X& operator=(const X&&) = delete;  // no move assignment
}

// initilizer list
std::vector<int>
    array = {1, 2, 3};
std::map<string, int> map = {{"a" : 1}, {"b" : 2}};

// direct list initialization:
T object{arg1, arg2, ...};
// copy list initialization:
T object = {arg1, arg2, ...};

// use list initialization to specify constructor of base class
class Y : X(10) {
}

// Constructor runtime behavior
template <typename T>
void swap(T& a, T& b) {
  T tmp{a};  // copy constructor
  a = b;     // copy assignment constructor
  b = tmp;   // copy assignment constructor
}

template <typename T>
void swap(T& a, T& b) {
  T temp{std::move(a)};  // move constructor
  a = std::move(b);      // move assignment constructor
  b = std::move(tmp);    // move assignment constructor
}

// Virtual
class V {
  // = 0 makes V a pure virtual, need to derive and implement
  V(const V&) = 0;

  // const member function
  //    members are read-only
  void mem_func() {
    cout << class_member << endl;
    class_member = 10;  // compile error
  }

  ///////////////////////
  // POINTERS/REFERENCE
  ////////////////////////

  // https://medium.com/@samathy_barratt/constantly-confusing-c-const-and-constexpr-pointer-behaviour-7eddb041e15a
  // https://medium.com/@ruturaj1989/pointers-and-references-part-2-when-and-how-to-use-them-cec046595761

  // pass by reference whenever possible; pass by pointer when needed
  void func(int& number) {}

  ////////////////////
  //    FUNCTION
  /////////////////////

  // inline function
  inline void display() {
    std::cout << "small function, benefit from no overhead of function call "
              << std::endl;
  }

// MACRO
#define FRUIT "text expand"

  ////////////////////
  //    TEMPLAE
  /////////////////////

  // https://github.com/wuye9036/CppTemplateTutorial

  // Template Class declaration
  template <typename T>
  class ClassA;

  // Template Class definition
  template <typename T>
  class ClassA {
    T member;
  };

  // Template Function delaration
  template <typename T>
  void foo(T const& v);
  template <typename T>
  T foo();
  template <typename T, typename U>
  U foo(T const&);

  // Template Function definition
  template <typename T>
  void foo() {
    T var;
    // ...
  }

  // multiple arguments return type deduction
  template <typename A, typename B, typename C>
  auto func(A a, B b, C c) -> decltype(a + b + c) {
    return a + b + c;
  }

  std::cout << func(5, 3.3, 1) << std::endl;
  // 回傳值自動推斷為 double
  std::cout << func(1, 0, 87) << std::endl;
  // 回傳值自動推斷為 int
  std::cout << func(5, 10, 6666666666666ll) << std::endl;
  // 回傳值自動推斷為 long long int

  ////////////////////
  //     OLD C++
  /////////////////////

  // 1. throw() -> noexcept
  // 2. #define / enum -> constexpr

  // 3. typedef -> using
  // struct v.s. typedef
  // https://stackoverflow.com/questions/612328/difference-between-struct-and-typedef-struct-in-c
  // typedef [original-type] [your-alias];
  typedef Map std::map<int, std::vector<int>>;

  // using [your-alias] = [original-type];
  using Map = std::map<int, std::vector<int>>;
  template <typename T1, typename T2>
  using Map = std::map<T1, std::vector<T2>>;

  // 4. type_list<type_list<type_list<... -> typename ...
  // 5. 私有建构 -> = delete
  // 6. 0 / NULL -> nullptr
