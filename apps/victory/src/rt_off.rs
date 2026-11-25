//! `VICTORY_RT_OFF` (on unless set to `0`): the title's ray-tracing passes
//! are not run. Their shaders are found by name in the package's shader
//! file and handed to the GPU layer, which drops every pipeline made with
//! one of them (not made on the host, its draws and dispatches not run).
//!
//! The passes: the traced shadows (with the indirect arguments they set up,
//! so their consumers go together), the traced visibility buffer, each
//! material's ray-tracing resolve, the screen probes' traces through the
//! structure, the deferred fallback through it, the checkerboarded traced
//! shadow masks and the particles' traced collisions. The cascaded shadows'
//! classification is kept: it runs whether tracing is on or not.

use std::collections::HashSet;
use std::path::Path;

use victory_game::sgd5;

/// Entry points (the part of a record's name after `family:`) of the
/// ray-tracing passes.
const ENTRIES: [&str; 8] = [
    "CS_RTShadowTrace",
    "CS_RTDistantShadow",
    "CS_SetRTShadowsIndirectArgs",
    "CS_RaytraceVBuffer",
    "CS_RaytracingResolve",
    "CS_DeferredFallback_BVH",
    "CS_GenerateShadowMasks_RaytracedCheckerboard",
    "CS_CollisionRaytraced",
];

/// Whether the record named `name` (`family:entry_<hash>_Wrapped`) is one
/// of a ray-tracing pass.
fn traces(name: &str) -> bool {
    if name.starts_with("cascadeshadows_classify:") {
        return false;
    }
    let entry = name.split_once(':').map_or(name, |(_, e)| e);
    ENTRIES.iter().any(|e| entry.starts_with(e))
        || (entry.starts_with("CS_ScreenProbes_TraceRays_") && entry.contains("BVH"))
}

fn on() -> bool {
    std::env::var_os("VICTORY_RT_OFF").is_none_or(|v| v != "0")
}

/// The shader hashes of the ray-tracing passes in an SGD5 file.
fn hashes(file: &[u8]) -> HashSet<String> {
    sgd5::records(file)
        .filter(|r| r.name.is_some_and(traces))
        .filter_map(|r| r.unpack())
        .map(|bytes| blake3::hash(&bytes).to_hex()[..16].to_owned())
        .collect()
}

/// With `VICTORY_RT_OFF` on, hands the ray-tracing shaders of the title in
/// `title_dir` to the GPU layer to drop.
pub(crate) fn apply(title_dir: &Path) {
    if !on() {
        println!("ray tracing    on (VICTORY_RT_OFF=0)");
        return;
    }
    let path = title_dir.join("shaders.rpf");
    let started = std::time::Instant::now();
    match std::fs::read(&path) {
        Ok(file) => {
            let found = hashes(&file);
            println!(
                "ray tracing    off: {} shaders of its passes dropped, found in {} in {:.1} s (VICTORY_RT_OFF=0 keeps it)",
                found.len(),
                path.display(),
                started.elapsed().as_secs_f64()
            );
            victory_gpu::pipeline::drop_shaders(found);
        }
        Err(e) => println!("ray tracing    left on: {} not read ({e})", path.display()),
    }
}

#[cfg(test)]
mod tests {
    use super::traces;

    #[test]
    fn names() {
        assert!(traces("sr_tracing:CS_RaytraceVBuffer_4f16bf6c_Wrapped"));
        assert!(traces("trees:CS_RaytracingResolve_01234567_Wrapped"));
        assert!(traces("indirect_diffuse_lighting:CS_ScreenProbes_TraceRays_BVH_01234567_Wrapped"));
        assert!(!traces("indirect_diffuse_lighting:CS_ScreenProbes_TraceRays_Screen_01234567_Wrapped"));
        assert!(!traces("cascadeshadows_classify:CS_RTShadowTrace_01234567_Wrapped"));
        assert!(!traces("cascadeshadows_classify:CS_ShadowUnifiedClearRT_45390227_Wrapped"));
        assert!(!traces("VS_AlphaTransmissionFullScreenTri_4f16bf6c_Wrapped"));
    }
}
