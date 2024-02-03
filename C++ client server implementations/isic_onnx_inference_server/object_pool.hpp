/// object_pool.hpp

#include <functional>
#include <iostream>
#include <memory>
#include <stack>
#include <tuple>
#include <mutex>


// The Object Pool
template <typename T, typename... Args> class ObjectPool {
public:
    template <typename P> using pointer_type = std::unique_ptr<P, std::function<void(P*)>>;
    
    ObjectPool(std::size_t init_size = 0, std::size_t max_size = 10, Args&&... args)
        : _max_size{max_size}, _available{max_size}, _size{0}, _args{args...} {
        initialize(init_size);
    }

    pointer_type<T> get() {
        if (_pool.empty()) {
            if (_available == 0) {
                return nullptr;
            }
            add();
        }
        mutex.lock();
        --_available;
        auto inst = std::move(_pool.top());
        _pool.pop();
        mutex.unlock();
        return std::move(inst);
    }

    std::size_t free() { return _available; }
    std::size_t max_size() { return _max_size; }
    std::size_t size() { return _size; }
    bool empty() { return _pool.empty(); }

private:
    // Adds a new object to the pool
    void add(T* ptr = nullptr) {        
        if (ptr == nullptr) {
            ptr = create_with_params(std::index_sequence_for<Args...>());
            mutex.lock();
            ++_size;
            mutex.unlock();
        } else {
            mutex.lock();
            ++_available;
            mutex.unlock();
        }

        pointer_type<T> inst(ptr, [this](T* ptr) {
            // This is the custom deleter of the unique_ptr.
            // When the object is deleted in the callers context, it will be
            // returned back to the pool by utilizing the add function
            add(ptr);
        });
        
        mutex.lock();
        _pool.push(std::move(inst));
        mutex.unlock();
    }

    template <std::size_t... Is> T* create_with_params(const std::index_sequence<Is...>&) {
        return new T(std::get<Is>(_args)...);
    }

    // Initializes the pool
    void initialize(std::size_t init_size) {
        for (std::size_t i = 0; i < init_size; ++i) {
            add();
        }
    }

    std::mutex mutex; // concurrent usage
    std::size_t _max_size;
    std::size_t _available;
    std::size_t _size;
    std::stack<pointer_type<T>> _pool;
    std::tuple<Args...> _args;
};            
