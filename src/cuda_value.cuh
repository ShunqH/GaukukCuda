#pragma once 

#include <iostream>     // std::cerr, cout, endl 
#include <stdexcept>    // std::runtime_error

namespace Gaukuk{

template<class T>
class CValue{
public: 
    CValue(): deviPrt_(nullptr) { AllocateVal(); };
    explicit CValue(const T& num): deviPrt_(nullptr) { 
        AllocateVal(); 
        CopyFromHost(num); 
    };

    // rule of five 
    ~CValue() noexcept{ DeleteVal(); }
    // deep copy (forbid)
    CValue(const CValue& other) = delete;
    CValue& operator=(const CValue& other) = delete;
    // move 
    CValue(CValue&& other) noexcept : deviPrt_(other.deviPrt_) {
        other.deviPrt_ = nullptr;
    }
    CValue& operator=(CValue&& other) noexcept {
        if (this != &other) {
            DeleteVal();
            deviPrt_ = other.deviPrt_;
            other.deviPrt_ = nullptr;
        }
        return *this;
    }

    T* View() noexcept { return deviPrt_; }
    const T* View() const noexcept { return deviPrt_; }

    void CopyFromHost(const T hostVal); 
    void CopyToHost(T& hostVal) const; 

private:
    T* deviPrt_; 

    void AllocateVal(); 
    void DeleteVal() noexcept; 
}; 

template<class T>
void CValue<T>::CopyFromHost(const T hostVal){
    if (deviPrt_ == nullptr) {
        std::cerr << "CValue is not allocated in CValue::CopyFromHost" << std::endl;
        return;
    }
    CUDA_CHECK(cudaMemcpy(deviPrt_, &hostVal, sizeof(T), cudaMemcpyHostToDevice));
}

template<class T>
void CValue<T>::CopyToHost(T& hostVal) const{
    if (deviPrt_ == nullptr) {
        std::cerr << "CValue is not allocated in CValue::CopyToHost" << std::endl;
        return;
    }
    CUDA_CHECK(cudaMemcpy(&hostVal, deviPrt_, sizeof(T), cudaMemcpyDeviceToHost));
}

template<class T>
void CValue<T>::AllocateVal(){
    T* newVal = nullptr;
    cudaError_t err = cudaMalloc(&newVal, sizeof(T));
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    deviPrt_ = newVal; 
}

template<class T>
void CValue<T>::DeleteVal() noexcept{
    if (deviPrt_){
        cudaFree(deviPrt_); 
    }
    deviPrt_ = nullptr; 
}

}