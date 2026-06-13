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

#[cfg(not(target_family = "wasm"))]
use crate::error::Error;
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

/// A disk-backed cache storing one file per URL (native only).
///
/// Each entry is `[u32 LE url length][url bytes][data bytes]`, so a hash
/// collision on the filename degrades to a cache miss (the stored URL is
/// verified on read) rather than serving the wrong tile. Writes are atomic
/// (temp file + rename), so a crash mid-write never leaves a torn entry.
///
/// No TTL: rocktree data is epoch-versioned and the epoch is part of the URL,
/// so a superseded entry is simply never requested again. The cache shares the
/// `<cache dir>/veldera` root with the rest of the project (see
/// [`FilesystemCache::veldera`]) but keeps its own `rocktree` subdirectory and
/// its own type — nothing is shared with other caches but the root path.
///
/// I/O is synchronous (small reads/writes wrapped in ready futures, like
/// [`MemoryCache`]), keeping the crate runtime-agnostic.
#[cfg(not(target_family = "wasm"))]
#[derive(Debug, Clone)]
pub struct FilesystemCache {
    dir: std::path::PathBuf,
}

#[cfg(not(target_family = "wasm"))]
impl FilesystemCache {
    /// Create a cache storing its files directly in `dir`.
    #[must_use]
    pub fn new(dir: impl Into<std::path::PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// Create a cache under the shared project cache root,
    /// `<OS cache dir>/veldera/rocktree`. Returns `None` when the OS cache
    /// directory cannot be resolved.
    #[must_use]
    pub fn veldera() -> Option<Self> {
        Some(Self::new(dirs::cache_dir()?.join("veldera").join("rocktree")))
    }

    /// The on-disk path for a URL's entry.
    fn path_for(&self, url: &str) -> std::path::PathBuf {
        self.dir.join(format!("{:016x}", fnv1a(url)))
    }

    /// Read the entry at `path`, returning its data only if the stored URL
    /// matches `url` (guarding against the rare filename-hash collision).
    fn read_verified(path: &std::path::Path, url: &str) -> Result<Option<Vec<u8>>> {
        let bytes = match std::fs::read(path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => {
                return Err(Error::Cache {
                    operation: "read",
                    message: e.to_string(),
                });
            }
        };
        let Some((stored_url, data)) = split_entry(&bytes) else {
            // Truncated or foreign file: treat as a miss.
            return Ok(None);
        };
        if stored_url == url.as_bytes() {
            Ok(Some(data.to_vec()))
        } else {
            Ok(None)
        }
    }
}

#[cfg(not(target_family = "wasm"))]
impl Cache for FilesystemCache {
    fn get(&self, url: &str) -> GetFuture<'_> {
        let result = Self::read_verified(&self.path_for(url), url);
        Box::pin(async move { result })
    }

    fn put(&self, url: &str, data: Vec<u8>) -> CacheFuture<'_> {
        let path = self.path_for(url);
        let result = write_entry(&self.dir, &path, url, &data);
        Box::pin(async move { result })
    }

    fn contains(&self, url: &str) -> ContainsFuture<'_> {
        let result = Self::read_verified(&self.path_for(url), url).map(|d| d.is_some());
        Box::pin(async move { result })
    }

    fn remove(&self, url: &str) -> CacheFuture<'_> {
        let result = match std::fs::remove_file(self.path_for(url)) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(Error::Cache {
                operation: "remove",
                message: e.to_string(),
            }),
        };
        Box::pin(async move { result })
    }

    fn clear(&self) -> CacheFuture<'_> {
        let result = match std::fs::remove_dir_all(&self.dir) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(Error::Cache {
                operation: "clear",
                message: e.to_string(),
            }),
        };
        Box::pin(async move { result })
    }
}

/// Split a stored entry into its URL and data halves, or `None` if the buffer
/// is too short or its declared URL length overruns it.
#[cfg(not(target_family = "wasm"))]
fn split_entry(bytes: &[u8]) -> Option<(&[u8], &[u8])> {
    let (len_bytes, rest) = bytes.split_first_chunk::<4>()?;
    let url_len = u32::from_le_bytes(*len_bytes) as usize;
    if rest.len() < url_len {
        return None;
    }
    Some(rest.split_at(url_len))
}

/// Write a URL/data entry to `path` atomically (temp file + rename), creating
/// `dir` if needed.
#[cfg(not(target_family = "wasm"))]
fn write_entry(
    dir: &std::path::Path,
    path: &std::path::Path,
    url: &str,
    data: &[u8],
) -> Result<()> {
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NONCE: AtomicU64 = AtomicU64::new(0);

    std::fs::create_dir_all(dir).map_err(|e| Error::Cache {
        operation: "create dir",
        message: e.to_string(),
    })?;

    let tmp = path.with_extension(format!(
        "tmp.{}.{}",
        std::process::id(),
        NONCE.fetch_add(1, Ordering::Relaxed)
    ));
    let write = || -> std::io::Result<()> {
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(&(url.len() as u32).to_le_bytes())?;
        file.write_all(url.as_bytes())?;
        file.write_all(data)?;
        file.sync_all()?;
        std::fs::rename(&tmp, path)
    };
    write().map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        Error::Cache {
            operation: "write",
            message: e.to_string(),
        }
    })
}

/// FNV-1a 64-bit hash, used to derive a filesystem-safe filename from a URL.
/// Stable across runs (collisions are caught by the stored-URL check).
#[cfg(not(target_family = "wasm"))]
fn fnv1a(s: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in s.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
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

        // These caches do their work synchronously, so the future is ready
        // on the first poll.
        match f.as_mut().poll(&mut cx) {
            Poll::Ready(result) => result,
            Poll::Pending => panic!("Future unexpectedly pending"),
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

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn test_filesystem_cache_roundtrip_and_collision_guard() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "veldera_fscache_test_{}_{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let cache = FilesystemCache::new(&dir);

        // Miss, store, hit.
        assert_eq!(block_on(cache.get("https://x/a")).unwrap(), None);
        block_on(cache.put("https://x/a", vec![1, 2, 3])).unwrap();
        assert_eq!(block_on(cache.get("https://x/a")).unwrap(), Some(vec![1, 2, 3]));
        assert!(block_on(cache.contains("https://x/a")).unwrap());

        // A different URL that lands on the same file would be caught by the
        // stored-URL check; directly, distinct URLs simply don't collide here.
        block_on(cache.put("https://x/b", vec![9])).unwrap();
        assert_eq!(block_on(cache.get("https://x/b")).unwrap(), Some(vec![9]));
        assert_eq!(block_on(cache.get("https://x/a")).unwrap(), Some(vec![1, 2, 3]));

        // Forged collision: write an entry under a's filename but b's URL, and
        // confirm a read for a treats it as a miss rather than returning b.
        let path = cache.path_for("https://x/a");
        super::write_entry(&dir, &path, "https://x/b", &[7, 7]).unwrap();
        assert_eq!(block_on(cache.get("https://x/a")).unwrap(), None);

        block_on(cache.remove("https://x/b")).unwrap();
        assert!(!block_on(cache.contains("https://x/b")).unwrap());
        block_on(cache.clear()).unwrap();
        assert!(!dir.exists());
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
