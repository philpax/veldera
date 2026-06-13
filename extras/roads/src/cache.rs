//! Disk cache for fetched road way-sets (native only).
//!
//! Each entry is the JSON-serialized set of [`RoadWay`]s for one quantized
//! region cell, written atomically (temp file + rename) so a crash mid-write
//! never leaves a torn entry. Entries carry a fetch timestamp and expire after
//! [`TTL`], because OSM road data drifts slowly but is not epoch-versioned the
//! way rocktree tiles are.
//!
//! The cache shares the `<cache dir>/veldera` root with the rest of the project
//! (see [`RoadCache::veldera`]) but keeps its own `roads` subdirectory and its
//! own type — nothing is shared with the rocktree cache but the root path.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::{Error, GeoBbox, Result, RoadWay};

/// The cell size used to quantize a region into a cache key, in degrees.
///
/// A region's southwest corner is floored to a multiple of this size, so all
/// requests landing in the same ≈0.02° cell (roughly 2 km of latitude) share a
/// cache entry. Callers requesting aligned cells get exact hits; this is a
/// coarse key, not a spatial index, so an overlapping-but-unaligned request is
/// a miss rather than a partial hit.
const CELL_DEGREES: f64 = 0.02;

/// How long a cached entry stays valid before it is treated as a miss.
const TTL: Duration = Duration::from_secs(4 * 7 * 24 * 60 * 60);

/// A disk-backed cache of fetched road way-sets, keyed by quantized region cell
/// (native only).
#[derive(Debug, Clone)]
pub struct RoadCache {
    dir: std::path::PathBuf,
    ttl: Duration,
}

impl RoadCache {
    /// Create a cache storing its files directly in `dir`, with the default
    /// [`TTL`].
    #[must_use]
    pub fn new(dir: impl Into<std::path::PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            ttl: TTL,
        }
    }

    /// Create a cache under the shared project cache root,
    /// `<OS cache dir>/veldera/roads`. Returns `None` when the OS cache
    /// directory cannot be resolved.
    #[must_use]
    pub fn veldera() -> Option<Self> {
        Some(Self::new(dirs::cache_dir()?.join("veldera").join("roads")))
    }

    /// Override the time-to-live (mainly for testing expiry).
    #[must_use]
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = ttl;
        self
    }

    /// Fetch the cached way-set for `region`'s cell, or `None` on a miss or an
    /// expired entry.
    pub fn get(&self, region: GeoBbox) -> Result<Option<Vec<RoadWay>>> {
        let path = self.path_for(region);
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => {
                return Err(Error::Cache {
                    operation: "read",
                    message: e.to_string(),
                });
            }
        };

        let entry: CacheEntry = match serde_json::from_slice(&bytes) {
            Ok(entry) => entry,
            // A corrupt or foreign file degrades to a miss.
            Err(_) => return Ok(None),
        };

        if entry.is_expired(self.ttl) {
            return Ok(None);
        }
        Ok(Some(entry.ways))
    }

    /// Store `ways` for `region`'s cell, stamped with the current time.
    pub fn put(&self, region: GeoBbox, ways: &[RoadWay]) -> Result<()> {
        let entry = CacheEntry {
            fetched_at_unix_secs: now_unix_secs(),
            ways: ways.to_vec(),
        };
        let bytes = serde_json::to_vec(&entry).map_err(|e| Error::Cache {
            operation: "serialize",
            message: e.to_string(),
        })?;
        write_atomic(&self.dir, &self.path_for(region), &bytes)
    }

    /// The on-disk path for a region's cell entry.
    fn path_for(&self, region: GeoBbox) -> std::path::PathBuf {
        let (cell_lat, cell_lon) = cell_key(region);
        self.dir.join(format!("{cell_lat}_{cell_lon}.json"))
    }
}

/// Quantize a region to its cell key: the integer multiples of [`CELL_DEGREES`]
/// at or below the southwest corner. Returned as signed cell indices so the key
/// is filename-safe and reversible.
fn cell_key(region: GeoBbox) -> (i64, i64) {
    let lat = (region.south / CELL_DEGREES).floor() as i64;
    let lon = (region.west / CELL_DEGREES).floor() as i64;
    (lat, lon)
}

/// A stored cache entry: the way-set plus the time it was fetched.
#[derive(Debug, Serialize, Deserialize)]
struct CacheEntry {
    fetched_at_unix_secs: u64,
    ways: Vec<RoadWay>,
}

impl CacheEntry {
    /// Whether the entry has reached or exceeded `ttl` in age. The boundary is
    /// inclusive so a zero TTL expires every entry immediately.
    fn is_expired(&self, ttl: Duration) -> bool {
        now_unix_secs().saturating_sub(self.fetched_at_unix_secs) >= ttl.as_secs()
    }
}

/// The current Unix time in whole seconds, or `0` if the clock is before the
/// epoch (which a TTL check treats as maximally stale).
fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Write `bytes` to `path` atomically (temp file + rename), creating `dir` if
/// needed.
fn write_atomic(dir: &std::path::Path, path: &std::path::Path, bytes: &[u8]) -> Result<()> {
    use std::{
        io::Write,
        sync::atomic::{AtomicU64, Ordering},
    };
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
        file.write_all(bytes)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LatLon, RoadClass};

    fn temp_dir() -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "veldera_roadcache_test_{}_{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn sample_way() -> RoadWay {
        RoadWay {
            node_ids: vec![1, 2],
            points: vec![
                LatLon {
                    lat: 40.7,
                    lon: -74.0,
                },
                LatLon {
                    lat: 40.71,
                    lon: -74.01,
                },
            ],
            class: RoadClass::Residential,
            bridge: false,
            tunnel: false,
            layer: 0,
            width: Some(3.5),
            lanes: Some(2.0),
        }
    }

    #[test]
    fn cell_key_floors_to_the_grid() {
        // 40.715 / 0.02 = 2035.75 -> floor 2035; -74.013 / 0.02 = -3700.65 ->
        // floor -3701.
        assert_eq!(
            cell_key(GeoBbox::new(40.715, -74.013, 40.74, -74.0)),
            (2035, -3701)
        );
        // Two regions sharing the same southwest cell map to the same key.
        assert_eq!(
            cell_key(GeoBbox::new(40.710, -74.019, 41.0, -73.0)),
            cell_key(GeoBbox::new(40.719, -74.001, 40.8, -73.5))
        );
    }

    #[test]
    fn roundtrip_and_ttl_expiry() {
        let dir = temp_dir();
        let region = GeoBbox::new(40.71, -74.05, 40.73, -74.03);
        let ways = vec![sample_way()];

        // Fresh cache (long TTL): miss, store, hit.
        let cache = RoadCache::new(&dir);
        assert_eq!(cache.get(region).unwrap(), None);
        cache.put(region, &ways).unwrap();
        assert_eq!(cache.get(region).unwrap(), Some(ways.clone()));

        // A zero TTL treats the just-written entry as expired.
        let expired = RoadCache::new(&dir).with_ttl(Duration::from_secs(0));
        assert_eq!(expired.get(region).unwrap(), None);

        // An unaligned, non-overlapping region is a separate cell and misses.
        let other = GeoBbox::new(41.50, -73.00, 41.52, -72.98);
        assert_eq!(cache.get(other).unwrap(), None);

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
