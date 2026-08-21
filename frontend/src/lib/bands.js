/**
 * The band boundaries, mirrored from backend/app/pipeline/smoothing.py.
 *
 * Kept in one place so the chart, the scale caption and anything else that
 * needs them cannot drift apart from each other. If they change on the
 * backend they must change here too - scripts/verify.py checks the backend
 * values, and these are what the interface draws.
 */
export const DRY_MAX = 0.25;
export const DAMP_MAX = 0.55;
