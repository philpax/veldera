//! Cache abstractions for storing fetched data.
//!
//! This module provides a `Cache` trait and implementations for caching
//! downloaded data to reduce network requests.
//!
//! # Implementations
//!
//! - [`MemoryCache`]: In-memory cache with optional size limits
//! - [`FilesystemCache`]: Disk-based cache (native only)
//! - [`NoCache`]: Passthrough implementation that caches nothing

use crate::error::Result;
use std::{
    collections::HashMap,
    future::Future,
    pin::Pin,
    sync::{Arc, RwLock},
};

/// Future type for cache get operations.
pub type GetFuture<'a> = Pin<Box<dyn Future<Output = Result<Option<Vec<u8>>>> + Send + 'a>>;

/// Future type for cache put/remove operations.
pub type CacheFuture<'a> = Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>>;

/// Future type for cache contains operations.
pub type ContainsFuture<'a> = Pin<Box<dyn Future<Output = Result<bool>> + Send + 'a>>;

/// A cache for storing fetched data.
///
/// The cache is keyed by URL and stores raw bytes. Implementations may
/// choose to store data in memory, on disk, or in any other persistent
/// storage.
pub trait Cache: Send + Sync {
    /// Get data from the cache.
    ///
    /// Returns `Ok(Some(data))` if the data is cached, `Ok(None)` if not cached,
    /// or an error if the cache operation failed.
    fn get(&self, url: &str) -> GetFuture<'_>;

    /// Store data in the cache.
    ///
    /// The data is associated with the given URL for later retrieval.
    fn put(&self, url: &str, data: Vec<u8>) -> CacheFuture<'_>;

    /// Check if data exists in the cache without retrieving it.
    fn contains(&self, url: &str) -> ContainsFuture<'_>;

    /// Remove data from the cache.
    fn remove(&self, url: &str) -> CacheFuture<'_>;

    /// Clear all cached data.
    fn clear(&self) -> CacheFuture<'_>;
}

/// A cache that stores nothing (passthrough).
///
/// This is useful when caching is not desired or for testing.
#[derive(Debug, Clone, Default)]
pub struct NoCache;

impl NoCache {
    /// Create a new no-op cache.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Cache for NoCache {
    fn get(&self, _url: &str) -> GetFuture<'_> {
        Box::pin(async { Ok(None) })
    }

    fn put(&self, _url: &str, _data: Vec<u8>) -> CacheFuture<'_> {
        Box::pin(async { Ok(()) })
    }

    fn contains(&self, _url: &str) -> ContainsFuture<'_> {
        Box::pin(async { Ok(false) })
    }

    fn remove(&self, _url: &str) -> CacheFuture<'_> {
        Box::pin(async { Ok(()) })
    }

    fn clear(&self) -> CacheFuture<'_> {
        Box::pin(async { Ok(()) })
    }
}

/// An in-memory cache.
///
/// This cache stores data in a `HashMap` protected by a `RwLock`. It's
/// suitable for short-lived applications or when disk caching is not needed.
///
/// The cache has an optional maximum size in bytes. When the limit is exceeded,
/// the oldest entries are evicted.
#[derive(Debug)]
pub struct MemoryCache {
    data: Arc<RwLock<MemoryCacheInner>>,
    max_size: Option<usize>,
}

#[derive(Debug, Default)]
struct MemoryCacheInner {
    entries: HashMap<String, Vec<u8>>,
    /// Insertion order for LRU eviction.
    order: Vec<String>,
    current_size: usize,
}

impl MemoryCache {
    /// Create a new memory cache with no size limit.
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: Arc::new(RwLock::new(MemoryCacheInner::default())),
            max_size: None,
        }
    }

    /// Create a new memory cache with a maximum size in bytes.
    #[must_use]
    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            data: Arc::new(RwLock::new(MemoryCacheInner::default())),
            max_size: Some(max_size),
        }
    }

    /// Get the current size of cached data in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.data.read().unwrap().current_size
    }

    /// Get the number of cached entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.read().unwrap().entries.len()
    }

    /// Check if the cache is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for MemoryCache {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for MemoryCache {
    fn clone(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
            max_size: self.max_size,
        }
    }
}

impl Cache for MemoryCache {
    fn get(&self, url: &str) -> GetFuture<'_> {
        let data = self.data.read().unwrap();
        let result = data.entries.get(url).cloned();
        Box::pin(async move { Ok(result) })
    }

    fn put(&self, url: &str, data: Vec<u8>) -> CacheFuture<'_> {
        let url = url.to_string();
        let mut cache = self.data.write().unwrap();

        // If the entry already exists, remove it first.
        if let Some(old_data) = cache.entries.remove(&url) {
            cache.current_size -= old_data.len();
            cache.order.retain(|k| k != &url);
        }

        let data_size = data.len();

        // Evict old entries if we have a size limit.
        if let Some(max_size) = self.max_size {
            while cache.current_size + data_size > max_size && !cache.order.is_empty() {
                let oldest = cache.order.remove(0);
                if let Some(old_data) = cache.entries.remove(&oldest) {
                    cache.current_size -= old_data.len();
                }
            }
        }

        cache.entries.insert(url.clone(), data);
        cache.order.push(url);
        cache.current_size += data_size;

        Box::pin(async { Ok(()) })
    }

    fn contains(&self, url: &str) -> ContainsFuture<'_> {
        let data = self.data.read().unwrap();
        let result = data.entries.contains_key(url);
        Box::pin(async move { Ok(result) })
    }

    fn remove(&self, url: &str) -> CacheFuture<'_> {
        let mut cache = self.data.write().unwrap();
        if let Some(data) = cache.entries.remove(url) {
            cache.current_size -= data.len();
            cache.order.retain(|k| k != url);
        }
        Box::pin(async { Ok(()) })
    }

    fn clear(&self) -> CacheFuture<'_> {
        let mut cache = self.data.write().unwrap();
        cache.entries.clear();
        cache.order.clear();
        cache.current_size = 0;
        Box::pin(async { Ok(()) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block_on<F: Future>(f: F) -> F::Output {
        // Simple polling executor for tests.
        use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

        fn dummy_raw_waker() -> RawWaker {
            fn no_op(_: *const ()) {}
            fn clone(_: *const ()) -> RawWaker {
                dummy_raw_waker()
            }
            static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, no_op, no_op, no_op);
            RawWaker::new(std::ptr::null(), &VTABLE)
        }

        #[allow(unsafe_code)]
        let waker = unsafe { Waker::from_raw(dummy_raw_waker()) };
        let mut cx = Context::from_waker(&waker);
        let mut f = std::pin::pin!(f);

        loop {
            match f.as_mut().poll(&mut cx) {
                Poll::Ready(result) => return result,
                Poll::Pending => {
                    // For these simple futures, they should always be ready.
                    panic!("Future unexpectedly pending");
                }
            }
        }
    }

    #[test]
    fn test_no_cache() {
        let cache = NoCache::new();

        // Put should succeed but not store anything.
        block_on(cache.put("http://example.com", vec![1, 2, 3])).unwrap();

        // Get should return None.
        let result = block_on(cache.get("http://example.com")).unwrap();
        assert!(result.is_none());

        // Contains should return false.
        let contains = block_on(cache.contains("http://example.com")).unwrap();
        assert!(!contains);
    }

    #[test]
    fn test_memory_cache_basic() {
        let cache = MemoryCache::new();

        // Initially empty.
        assert!(cache.is_empty());
        assert_eq!(cache.size(), 0);

        // Put data.
        block_on(cache.put("http://example.com/a", vec![1, 2, 3])).unwrap();
        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.size(), 3);

        // Get data.
        let result = block_on(cache.get("http://example.com/a")).unwrap();
        assert_eq!(result, Some(vec![1, 2, 3]));

        // Contains.
        assert!(block_on(cache.contains("http://example.com/a")).unwrap());
        assert!(!block_on(cache.contains("http://example.com/b")).unwrap());

        // Remove.
        block_on(cache.remove("http://example.com/a")).unwrap();
        assert!(cache.is_empty());
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_memory_cache_eviction() {
        // Cache with 10-byte limit.
        let cache = MemoryCache::with_max_size(10);

        // Add 5 bytes.
        block_on(cache.put("http://a", vec![1, 2, 3, 4, 5])).unwrap();
        assert_eq!(cache.size(), 5);
        assert!(block_on(cache.contains("http://a")).unwrap());

        // Add 5 more bytes.
        block_on(cache.put("http://b", vec![6, 7, 8, 9, 10])).unwrap();
        assert_eq!(cache.size(), 10);

        // Add 3 more bytes, which should evict "http://a".
        block_on(cache.put("http://c", vec![11, 12, 13])).unwrap();
        assert_eq!(cache.size(), 8); // 5 + 3.
        assert!(!block_on(cache.contains("http://a")).unwrap());
        assert!(block_on(cache.contains("http://b")).unwrap());
        assert!(block_on(cache.contains("http://c")).unwrap());
    }

    #[test]
    fn test_memory_cache_clear() {
        let cache = MemoryCache::new();

        block_on(cache.put("http://a", vec![1, 2, 3])).unwrap();
        block_on(cache.put("http://b", vec![4, 5, 6])).unwrap();
        assert_eq!(cache.len(), 2);

        block_on(cache.clear()).unwrap();
        assert!(cache.is_empty());
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_memory_cache_update() {
        let cache = MemoryCache::new();

        // Add initial data.
        block_on(cache.put("http://a", vec![1, 2, 3])).unwrap();
        assert_eq!(cache.size(), 3);

        // Update with larger data.
        block_on(cache.put("http://a", vec![1, 2, 3, 4, 5])).unwrap();
        assert_eq!(cache.size(), 5);
        assert_eq!(cache.len(), 1);

        let result = block_on(cache.get("http://a")).unwrap();
        assert_eq!(result, Some(vec![1, 2, 3, 4, 5]));
    }
}
