#pragma once 

// C++ headers
#include <cstddef>  // size_t
#include <algorithm> // swap, copy

namespace Gaukuk{

template<class T>
class TArray{
public:
    enum class ArrayStatus { empty, allocated }; 
    TArray() : pdata_(nullptr), n1_(0), n2_(0), n3_(0), n4_(0), nArr(0), state_(ArrayStatus::empty){}; 
    explicit TArray(int n1) : pdata_(nullptr), 
                                 n1_(n1), n2_(1), n3_(1), n4_(1), nArr(0), 
                                 state_(ArrayStatus::empty) { AllocateArray(); }; 
    TArray(int n2, int n1) : pdata_(nullptr), 
                                 n1_(n1), n2_(n2), n3_(1), n4_(1), nArr(0), 
                                 state_(ArrayStatus::empty) { AllocateArray(); }; 
    TArray(int n3, int n2, int n1) : pdata_(nullptr), 
                                 n1_(n1), n2_(n2), n3_(n3), n4_(1), nArr(0), 
                                 state_(ArrayStatus::empty) { AllocateArray(); }; 
    TArray(int n4, int n3, int n2, int n1) : pdata_(nullptr), 
                                 n1_(n1), n2_(n2), n3_(n3), n4_(n4), nArr(0), 
                                 state_(ArrayStatus::empty) { AllocateArray(); }; 
    
    // rule of five 
    ~TArray() noexcept; 
    // deep copy 
    TArray(const TArray<T>& other); 
    TArray<T>& operator=(const TArray<T>& other); 
    // move 
    TArray(TArray<T>&& other) noexcept; 
    TArray<T>& operator=(TArray<T>&& other) noexcept; 

    // method to allocate new array
    void NewArray(int n1); 
    void NewArray(int n2, int n1); 
    void NewArray(int n3, int n2, int n1); 
    void NewArray(int n4, int n3, int n2, int n1); 

    size_t GetSize() const { return nArr; }
    size_t GetSizeInBytes() const { return nArr*sizeof(T); }
    int GetN1() const { return n1_; }
    int GetN2() const { return n2_; }
    int GetN3() const { return n3_; }
    int GetN4() const { return n4_; }

    T& operator()(const int i) {
        return pdata_[i]; 
    }
    T operator()(const int i) const {
        return pdata_[i]; 
    }
    T& operator()(const int n, const int i) {
        return pdata_[i + n1_*n]; 
    }
    T operator()(const int n, const int i) const {
        return pdata_[i + n1_*n]; 
    }
    T& operator()(const int n, const int j, const int i) {
        return pdata_[i + n1_*(j + n2_*n)]; 
    }
    T operator()(const int n, const int j, const int i) const {
        return pdata_[i + n1_*(j + n2_*n)]; 
    }
    T& operator()(const int n, const int k, const int j, const int i) {
        return pdata_[i + n1_*(j + n2_*(k + n3_*n))]; 
    }
    T operator()(const int n, const int k, const int j, const int i) const {
        return pdata_[i + n1_*(j + n2_*(k + n3_*n))]; 
    }

    T* data() noexcept { return pdata_; }
    const T* data() const noexcept { return pdata_; }

    void Swap(TArray& other) noexcept; 
    void DeleteArray() noexcept; 
private:
    T* pdata_; 
    int n1_, n2_, n3_, n4_;
    size_t nArr; 
    ArrayStatus state_; 

    void AllocateArray();
}; 

template<class T>
TArray<T>::~TArray() noexcept{
    DeleteArray(); 
}

// deep copy constructor
template<class T>
TArray<T>::TArray(const TArray<T>& other) : 
    n1_(other.n1_),
    n2_(other.n2_),
    n3_(other.n3_),
    n4_(other.n4_),
    nArr(other.nArr),
    state_(other.state_),
    pdata_(nullptr)     
{
    if (other.pdata_) {
        AllocateArray();   
        std::copy(other.pdata_, other.pdata_ + other.nArr, pdata_);
    }
}

// deep copy assignment operator
template<class T>
TArray<T>& TArray<T>::operator=(const TArray<T>& other){
    if (this != &other){
        TArray<T> newArr(other); 
        Swap(newArr); 
    }
    return *this; 
}

// move constructor
template<class T> 
TArray<T>::TArray(TArray<T>&& other) noexcept{
    n1_ = other.n1_; 
    n2_ = other.n2_; 
    n3_ = other.n3_; 
    n4_ = other.n4_; 
    nArr = other.nArr; 
    pdata_ = other.pdata_; 
    state_ = other.state_; 
    other.n1_ = 0; 
    other.n2_ = 0; 
    other.n3_ = 0; 
    other.n4_ = 0; 
    other.nArr = 0; 
    other.pdata_ = nullptr; 
    other.state_ = ArrayStatus::empty; 
}

// move assignment operator
template<class T>
TArray<T>& TArray<T>::operator=(TArray<T>&& other) noexcept{
    if (this != &other){
        DeleteArray(); 
        n1_ = other.n1_; 
        n2_ = other.n2_; 
        n3_ = other.n3_; 
        n4_ = other.n4_; 
        nArr = other.nArr; 
        pdata_ = other.pdata_; 
        state_ = other.state_; 
        other.n1_ = 0; 
        other.n2_ = 0; 
        other.n3_ = 0; 
        other.n4_ = 0; 
        other.nArr = 0; 
        other.pdata_ = nullptr; 
        other.state_ = ArrayStatus::empty; 
    }
    return *this; 
}

template<class T>
void TArray<T>::NewArray(int n1){
    TArray<T> newArr(n1); 
    Swap(newArr); 
}

template<class T>
void TArray<T>::NewArray(int n2, int n1){
    TArray<T> newArr(n2, n1); 
    Swap(newArr); 
}

template<class T>
void TArray<T>::NewArray(int n3, int n2, int n1){
    TArray<T> newArr(n3, n2, n1); 
    Swap(newArr); 
}

template<class T>
void TArray<T>::NewArray(int n4, int n3, int n2, int n1){
    TArray<T> newArr(n4, n3, n2, n1); 
    Swap(newArr); 
}

template<class T>
void TArray<T>::Swap(TArray<T>& other) noexcept {
    using std::swap;
    swap(n1_, other.n1_);
    swap(n2_, other.n2_);
    swap(n3_, other.n3_);
    swap(n4_, other.n4_);
    swap(nArr, other.nArr);
    swap(pdata_, other.pdata_);
    swap(state_, other.state_);
}

template<class T>
void TArray<T>::AllocateArray(){
    nArr = n1_*n2_*n3_*n4_; 
    pdata_ = new T[nArr](); 
    state_ = ArrayStatus::allocated; 
}

template<class T>
void TArray<T>::DeleteArray() noexcept{
    if (state_ == ArrayStatus::allocated){
        delete[] pdata_; 
        nArr = 0; 
        state_ = ArrayStatus::empty; 
    }
    pdata_ = nullptr; 
}

} // namespace Gaukuk
