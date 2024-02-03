#include <memory>
#include <stack>
#include <stdexcept>
#include <mutex>

template <class T, class D = std::default_delete<T>>
class SmartObjectPool
{
 private:
  std::mutex mutex;  // make the class thread safe

  struct ReturnToPool_Deleter {
    explicit ReturnToPool_Deleter(std::weak_ptr<SmartObjectPool<T, D>* > pool)
        : pool_(pool) {}

    void operator()(T* ptr) {
      mutex.lock();
      if (auto pool_ptr = pool_.lock())
          (*pool_ptr.get())->add(std::unique_ptr<T, D>{ptr});
      else
          D{}(ptr);
      mutex.unlock();
    }
   private:
    std::weak_ptr<SmartObjectPool<T, D>* > pool_;
  };

 public:
  using ptr_type = std::unique_ptr<T, ReturnToPool_Deleter >;

  SmartObjectPool() : this_ptr_(new SmartObjectPool<T, D>*(this)) {}
  virtual ~SmartObjectPool(){}

  void add(std::unique_ptr<T, D> t) {
    mutex.lock();
    pool_.push(std::move(t));
    mutex.unlock();
  }

  ptr_type acquire() {
    if (pool_.empty())
      //throw std::out_of_range("Cannot acquire object from an empty pool.");
      return ptr_type(nullptr);

    mutex.lock();
    ptr_type tmp(pool_.top().release(),
                 ReturnToPool_Deleter{
                   std::weak_ptr<SmartObjectPool<T, D>*>{this_ptr_}});
    pool_.pop();
    mutex.unlock();
    return std::move(tmp);
  }

  bool empty() const {
    return pool_.empty();
  }

  size_t size() const {
    return pool_.size();
  }

 private:
  std::shared_ptr<SmartObjectPool<T, D>* > this_ptr_;
  std::stack<std::unique_ptr<T, D> > pool_;
};

