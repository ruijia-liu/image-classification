function summary = run_pipeline(stepName)
% Adapted from the original analysis implementation; identifiers and I/O generalized.
%RUN_PIPELINE Simple controller for the image multispectral analysis.
%
% Examples:
%   run_pipeline("inspect")
%   run_pipeline("fast_all")

if nargin < 1 || strlength(string(stepName)) == 0
    error("run_pipeline:MissingStep", ...
        "Provide a pipeline step, for example run_pipeline(""fast_all""). Supported steps: %s", ...
        strjoin(supportedSteps(), ", "));
end

stepName = lower(strtrim(string(stepName)));
cfg = config();
ensureOutputDirectories(cfg);

logFile = fullfile(cfg.paths.logs, "pipeline_master_log.txt");
pipelineStart = datetime("now");
pipelineTimer = tic;

fid = fopen(logFile, "a");
if fid < 0
    error("run_pipeline:LogFile", "Cannot open pipeline log file: %s", logFile);
end
cleanupLog = onCleanup(@() fclose(fid));

logLine(fid, "");
logLine(fid, "============================================================");
logLine(fid, "Pipeline started: %s", string(pipelineStart));
logLine(fid, "Requested step: %s", stepName);
logLine(fid, "FAST_MODE: %d", logicalValue(cfg, "FAST_MODE"));
logLine(fid, "Parallel pool auto-start: disabled by pipeline");

fprintf("Image analysis pipeline\n");
fprintf("Step: %s\n", stepName);
fprintf("Master log: %s\n\n", logFile);

knownFilesBefore = snapshotGeneratedFiles(cfg);
stepRows = table();

try
    switch stepName
        case "inspect"
            stepRows = runOneStep(stepRows, fid, "inspect", @inspect_mat_files, false);

        case "extract"
            stepRows = runOneStep(stepRows, fid, "extract", @extract_feature_tables, false);

        case "quality"
            requireCache(cfg);
            stepRows = runOneStep(stepRows, fid, "quality", @check_data_quality, false);

        case "spectra"
            requireObservationCache(cfg);
            stepRows = runOneStep(stepRows, fid, "spectra", @analyse_spectra, false);

        case "embedding"
            requireObservationCache(cfg);
            stepRows = runOneStep(stepRows, fid, "embedding", @analyse_pca_tsne, false);

        case "classify"
            requireObservationCache(cfg);
            stepRows = runOneStep(stepRows, fid, "classify", @classify_by_time, false);

        case "source"
            stepRows = runOneStep(stepRows, fid, "source", @evaluate_source_level, false);

        case "cross_domain"
            requireObservationCache(cfg);
            stepRows = runOneStep(stepRows, fid, "cross_domain", @validate_across_domains, false);

        case "cross_time"
            requireObservationCache(cfg);
            stepRows = runOneStep(stepRows, fid, "cross_time", @validate_across_time, false);

        case "importance"
            requireObservationCache(cfg);
            stepRows = runOneStep(stepRows, fid, "importance", @analyse_band_importance, false);

        case "trajectory"
            requireObservationCache(cfg);
            stepRows = runOneStep(stepRows, fid, "trajectory", @analyse_tsne_time_trajectory, false);

        case "fast_all"
            stepRows = runFastAll(stepRows, fid, cfg);

        otherwise
            error("run_pipeline:UnknownStep", ...
                "Unknown step ""%s"". Supported steps: %s", ...
                stepName, strjoin(supportedSteps(), ", "));
    end
catch ME
    logLine(fid, "PIPELINE FAILED: %s", ME.message);
    logLine(fid, "%s", getReport(ME, "extended", "hyperlinks", "off"));
    rethrow(ME);
end

elapsedSeconds = toc(pipelineTimer);
generatedFiles = newOrUpdatedFiles(cfg, knownFilesBefore);
summary = struct();
summary.step = stepName;
summary.started_at = pipelineStart;
summary.elapsed_seconds = elapsedSeconds;
summary.step_summary = stepRows;
summary.master_log = logFile;
summary.generated_files = generatedFiles;

summaryFile = fullfile(cfg.paths.logs, "pipeline_step_summary.csv");
writetable(stepRows, summaryFile);

logLine(fid, "Pipeline completed: %s", string(datetime("now")));
logLine(fid, "Total runtime seconds: %.2f", elapsedSeconds);
logGeneratedFiles(fid, generatedFiles);

fprintf("\nPipeline complete in %.2f seconds.\n", elapsedSeconds);
fprintf("Step summary saved to: %s\n", summaryFile);
printGeneratedFiles(generatedFiles);

end

function stepRows = runFastAll(stepRows, fid, cfg)
logLine(fid, "Running fast_all sequence.");

if cacheMissing(cfg)
    stepRows = runOneStep(stepRows, fid, "extract", @extract_feature_tables, false);
else
    logLine(fid, "Skipping extraction because cache files already exist.");
    fprintf("Cache found; skipping extraction.\n");
    stepRows = appendStepRow(stepRows, "extract", "skipped", ...
        "cache already exists", 0);
end

requireCache(cfg);

stepRows = runOneStep(stepRows, fid, "quality", @check_data_quality, true);
stepRows = runOneStep(stepRows, fid, "spectra", @analyse_spectra, true);
stepRows = runOneStep(stepRows, fid, "embedding", ...
    @() analyse_pca_tsne(struct('RUN_DOMAIN_TIME_TSNE', false, 'RUN_SLOW_ANALYSES', false)), true);

stepRows = runOneStep(stepRows, fid, "classify", @classify_by_time, false);
stepRows = runOneStep(stepRows, fid, "cross_domain", @validate_across_domains, false);
stepRows = runOneStep(stepRows, fid, "importance", @analyse_band_importance, true);

logLine(fid, "fast_all finished.");
end

function stepRows = runOneStep(stepRows, fid, name, functionHandle, optionalStep)
tStart = tic;
started = datetime("now");
logLine(fid, "STEP START: %s at %s", name, string(started));
fprintf("Running %s...\n", name);

try
    functionHandle();
    elapsed = toc(tStart);
    logLine(fid, "STEP OK: %s runtime %.2f seconds", name, elapsed);
    fprintf("  %s complete (%.2f seconds)\n", name, elapsed);
    stepRows = appendStepRow(stepRows, name, "completed", "", elapsed);
catch ME
    elapsed = toc(tStart);
    statusText = "failed";
    if optionalStep
        statusText = "optional_failed";
    end
    logLine(fid, "STEP %s: %s runtime %.2f seconds", upper(statusText), name, elapsed);
    logLine(fid, "ERROR: %s", ME.message);
    logLine(fid, "%s", getReport(ME, "extended", "hyperlinks", "off"));
    stepRows = appendStepRow(stepRows, name, statusText, ME.message, elapsed);

    if optionalStep
        warning("run_pipeline:OptionalStepFailed", ...
            "Optional step %s failed and was logged: %s", name, ME.message);
    else
        error("run_pipeline:StepFailed", ...
            "Pipeline step %s failed. See %s. Original error: %s", ...
            name, getLogPath(fid), ME.message);
    end
end
end

function stepRows = appendStepRow(stepRows, name, statusText, messageText, elapsedSeconds)
row = table( ...
    string(name), ...
    string(statusText), ...
    string(messageText), ...
    elapsedSeconds, ...
    string(datetime("now")), ...
    'VariableNames', {'step', 'status', 'message', 'runtime_seconds', 'finished_at'});
stepRows = [stepRows; row]; %#ok<AGROW>
end

function ensureOutputDirectories(cfg)
dirs = [
    string(cfg.paths.cache)
    string(cfg.paths.figures)
    string(cfg.paths.tables)
    string(cfg.paths.predictions)
    string(cfg.paths.logs)
    ];

for i = 1:numel(dirs)
    if ~isfolder(dirs(i))
        mkdir(dirs(i));
    end
end
end

function requireCache(cfg)
missing = cacheMissingList(cfg);
if ~isempty(missing)
    error("run_pipeline:MissingCache", ...
        "Cached tables are missing: %s. Run run_pipeline(""extract"") first.", ...
        strjoin(missing, ", "));
end
end

function requireObservationCache(cfg)
filename = fullfile(cfg.paths.cache, "observation_table.mat");
if ~isfile(filename)
    error("run_pipeline:MissingObservationCache", ...
        "Observation cache is missing: %s. Run run_pipeline(""extract"") first.", ...
        filename);
end
end

function tf = cacheMissing(cfg)
tf = ~isempty(cacheMissingList(cfg));
end

function missing = cacheMissingList(cfg)
files = [
    string(fullfile(cfg.paths.cache, "patch_table.mat"))
    string(fullfile(cfg.paths.cache, "observation_table.mat"))
    string(fullfile(cfg.paths.cache, "extraction_metadata.mat"))
    ];
missing = strings(0, 1);
for i = 1:numel(files)
    if ~isfile(files(i))
        missing(end + 1, 1) = files(i); %#ok<AGROW>
    end
end
end

function before = snapshotGeneratedFiles(cfg)
files = listGeneratedFiles(cfg);
before = containers.Map("KeyType", "char", "ValueType", "double");
for i = 1:numel(files)
    before(char(files(i).path)) = files(i).datenum;
end
end

function changed = newOrUpdatedFiles(cfg, before)
files = listGeneratedFiles(cfg);
rows = strings(0, 1);
for i = 1:numel(files)
    key = char(files(i).path);
    isNew = ~isKey(before, key);
    isUpdated = ~isNew && files(i).datenum > before(key) + 1e-9;
    if isNew || isUpdated
        rows(end + 1, 1) = files(i).path; %#ok<AGROW>
    end
end
changed = sort(rows);
end

function files = listGeneratedFiles(cfg)
roots = [
    string(cfg.paths.cache)
    string(cfg.paths.figures)
    string(cfg.paths.tables)
    string(cfg.paths.predictions)
    string(cfg.paths.logs)
    ];

files = struct('path', strings(0, 1), 'datenum', []);
idx = 0;
for r = 1:numel(roots)
    if ~isfolder(roots(r))
        continue
    end
    listing = dir(fullfile(roots(r), "**", "*"));
    for i = 1:numel(listing)
        if listing(i).isdir
            continue
        end
        idx = idx + 1;
        files(idx).path = string(fullfile(listing(i).folder, listing(i).name)); %#ok<AGROW>
        files(idx).datenum = listing(i).datenum; %#ok<AGROW>
    end
end
end

function printGeneratedFiles(files)
fprintf("\nGenerated or updated output files:\n");
if isempty(files)
    fprintf("  none detected\n");
    return
end

maxShown = min(numel(files), 30);
for i = 1:maxShown
    fprintf("  %s\n", files(i));
end
if numel(files) > maxShown
    fprintf("  ... %d more files\n", numel(files) - maxShown);
end
end

function logGeneratedFiles(fid, files)
logLine(fid, "Generated or updated output files:");
if isempty(files)
    logLine(fid, "  none detected");
    return
end
for i = 1:numel(files)
    logLine(fid, "  %s", files(i));
end
end

function steps = supportedSteps()
steps = [
    "inspect"
    "extract"
    "quality"
    "spectra"
    "embedding"
    "classify"
    "source"
    "cross_domain"
    "cross_time"
    "importance"
    "trajectory"
    "fast_all"
    ];
end

function value = logicalValue(cfg, fieldName)
if isfield(cfg, fieldName)
    value = logical(cfg.(fieldName));
else
    value = false;
end
end

function writeTextFile(filename, text)
fid = fopen(filename, "w");
if fid < 0
    error("run_pipeline:WriteFile", "Cannot write file: %s", filename);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, "%s", text);
clear cleanup
end

function logLine(fid, varargin)
message = sprintf(varargin{:});
fprintf(fid, "[%s] %s\n", string(datetime("now")), message);
end

function path = getLogPath(~)
path = fullfile(config().paths.logs, "pipeline_master_log.txt");
end
