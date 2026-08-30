#pragma once 

// C++ headers 
#include <iostream> // std::cerr, cout, endl 
#include <cstddef>  // size_t
#include <algorithm> // swap, copy 
#include <cuda_runtime.h> // cudaMalloc, cudaFree 
#include "./utils/utils_cuda.cuh"
#include "./template_array.hpp"

namespace Gaukuk{

template<class T>
struct CArrayView
{
    T* pdata = nullptr;
    int n1 = 0;
    int n2 = 0;
    int n3 = 0;
    int n4 = 0; 

    __host__ __device__ T& operator()(const int i) {
        return pdata[i]; 
    }
    __host__ __device__ const T& operator()(const int i) const {
        return pdata[i]; 
    }
    __host__ __device__ T& operator()(const int n, const int i) {
        return pdata[i + n1*n]; 
    }
    __host__ __device__ const T& operator()(const int n, const int i) const {
        return pdata[i + n1*n]; 
    }
    __host__ __device__ T& operator()(const int n, const int j, const int i) {
        return pdata[i + n1*(j + n2*n)]; 
    }
    __host__ __device__ const T& operator()(const int n, const int j, const int i) const {
        return pdata[i + n1*(j + n2*n)]; 
    }
    __host__ __device__ T& operator()(const int n, const int k, const int j, const int i) {
        return pdata[i + n1*(j + n2*(k + n3*n))]; 
    }
    __host__ __device__ const T& operator()(const int n, const int k, const int j, const int i) const {
        return pdata[i + n1*(j + n2*(k + n3*n))]; 
    }
};

template<class T>
class CArray{
public:
    enum class ArrayStatus { empty, allocated }; 
    CArray() : pdata_(nullptr), n1_(0), n2_(0), n3_(0), n4_(0), nArr(0), state_(ArrayStatus::empty){}; 
    explicit CArray(int n1) : pdata_(nullptr), 
                                 n1_(n1), n2_(1), n3_(1), n4_(1), nArr(0), 
                                 state_(ArrayStatus::empty) { AllocateArray(); }; 
    CArray(int n2, int n1) : pdata_(nullptr), 
                                 n1_(n1), n2_(n2), n3_(1), n4_(1), nArr(0), 
                                 state_(ArrayStatus::empty) { AllocateArray(); }; 
    CArray(int n3, int n2, int n1) : pdata_(nullptr), 
                                 n1_(n1), n2_(n2), n3_(n3), n4_(1), nArr(0), 
                                 state_(ArrayStatus::empty) { AllocateArray(); }; 
    CArray(int n4, int n3, int n2, int n1) : pdata_(nullptr), 
                                 n1_(n1), n2_(n2), n3_(n3), n4_(n4), nArr(0), 
                                 state_(ArrayStatus::empty) { AllocateArray(); }; 
                            
    // rule of five 
    ~CArray() noexcept; 
    // deep copy (forbid)
    CArray(const CArray<T>& other) = delete; 
    CArray<T>& operator=(const CArray<T>& other) = delete; 
    // move 
    CArray(CArray<T>&& other) noexcept; 
    CArray<T>& operator=(CArray<T>&& other) noexcept; 

        // method to allocate new array
    void NewArray(int n1); 
    void NewArray(int n2, int n1); 
    void NewArray(int n3, int n2, int n1); 
    void NewArray(int n4, int n3, int n2, int n1); 

    size_t GetSize() const { return nArr; }
    size_t GetSizeInBytes() const { return nArr*sizeof(T); }
    int GetN1() const noexcept { return n1_; }
    int GetN2() const noexcept { return n2_; }
    int GetN3() const noexcept { return n3_; }
    int GetN4() const noexcept { return n4_; }

    T* data() noexcept { return pdata_; }
    const T* data() const noexcept { return pdata_; }
    CArrayView<T> View() const{
        return {pdata_, n1_, n2_, n3_, n4_}; 
    }

    void Swap(CArray& other) noexcept; 
    void CopyFromHost(const TArray<T>& hostArr); 
    void CopyToHost(TArray<T>& hostArr) const; 

private:
    T* pdata_; 
    int n1_, n2_, n3_, n4_;
    size_t nArr; 
    ArrayStatus state_; 

    void AllocateArray();
    void DeleteArray() noexcept; 
}; 

template<class T>
CArray<T>::~CArray() noexcept{
    DeleteArray(); 
}

// move constructor
template<class T> 
CArray<T>::CArray(CArray<T>&& other) noexcept{
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
CArray<T>& CArray<T>::operator=(CArray<T>&& other) noexcept{
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
void CArray<T>::NewArray(int n1){
    CArray<T> newArr(n1); 
    Swap(newArr); 
}

template<class T>
void CArray<T>::NewArray(int n2, int n1){
    CArray<T> newArr(n2, n1); 
    Swap(newArr); 
}

template<class T>
void CArray<T>::NewArray(int n3, int n2, int n1){
    CArray<T> newArr(n3, n2, n1); 
    Swap(newArr); 
}

template<class T>
void CArray<T>::NewArray(int n4, int n3, int n2, int n1){
    CArray<T> newArr(n4, n3, n2, n1); 
    Swap(newArr); 
}

template<class T>
void CArray<T>::Swap(CArray<T>& other) noexcept {
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
void CArray<T>::CopyFromHost(const TArray<T>& hostArr){
    if (state_ != ArrayStatus::allocated || hostArr.GetSize() == 0) {
        std::cerr << "CArray not allocated in CopyFromHost" << std::endl;
        return;
    }
    size_t sizeInBytes = nArr*sizeof(T); 
    if (hostArr.GetSizeInBytes() == sizeInBytes){
        CUDA_CHECK(cudaMemcpy(pdata_, hostArr.data(), sizeInBytes, 
                              cudaMemcpyHostToDevice));
        return; 
    }
    std::cerr << "Array size doesn't match in CArray::CpFromTArray" << std::endl;
}

template<class T>
void CArray<T>::CopyToHost(TArray<T>& hostArr) const{
    size_t sizeInBytes = nArr*sizeof(T); 
    if (hostArr.GetSizeInBytes() == sizeInBytes){
        CUDA_CHECK(cudaMemcpy(hostArr.data(), pdata_, sizeInBytes, 
                              cudaMemcpyDeviceToHost));
        return; 
    }
    std::cerr << "Array size doesn't match in CArray::CpToTArray" << std::endl;
}

template<class T>
void CArray<T>::AllocateArray(){
    nArr = static_cast<size_t>(n1_) *
           static_cast<size_t>(n2_) *
           static_cast<size_t>(n3_) *
           static_cast<size_t>(n4_);
    if (nArr == 0) {
        pdata_ = nullptr;
        state_ = ArrayStatus::empty;
        return;
    }
    size_t bytes = nArr * sizeof(T); 
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pdata_), bytes)); 
    state_ = ArrayStatus::allocated; 
}

template<class T>
void CArray<T>::DeleteArray() noexcept{
    if (state_ == ArrayStatus::allocated){
        if (pdata_ != nullptr) cudaFree(pdata_);
        nArr = 0; 
        state_ = ArrayStatus::empty; 
    }
    pdata_ = nullptr; 
}

template<class T>
void TArrayToCArray(const TArray<T>& tarr, CArray<T>& carr){
    if (tarr.GetSizeInBytes()==carr.GetSizeInBytes()){
        CUDA_CHECK(cudaMemcpy(carr.data(), tarr.data(), tarr.GetSizeInBytes(), 
                              cudaMemcpyHostToDevice));
        return; 
    }
    std::cerr << "Array size doesn't match in TArrayToCArray" << std::endl;
}

template<class T>
void CArrayToTArray(const CArray<T>& carr, TArray<T>& tarr){
    if (carr.GetSizeInBytes()==tarr.GetSizeInBytes()){
        CUDA_CHECK(cudaMemcpy(tarr.data(), carr.data(), carr.GetSizeInBytes(), 
                              cudaMemcpyDeviceToHost));
        return; 
    }
    std::cerr << "Array size doesn't match in CArrayToTArray" << std::endl;
}

}