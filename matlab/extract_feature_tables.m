function summary = extract_feature_tables()
% Adapted from the original analysis implementation; identifiers and I/O generalized.
%EXTRACT_FEATURE_TABLES Extract lightweight feature tables from image MAT files.

tStart = tic;
cfg = config();
ensureOutputDirs(cfg);

warningFile = fullfile(cfg.paths.logs, 'extraction_warnings.txt');
warningFid = fopen(warningFile, 'w');
if warningFid < 0
    error('Could not open warning log for writing: %s', warningFile);
end
cleanup = onCleanup(@() fclose(warningFid)); %#ok<NASGU>

rng(cfg.RANDOM_SEED);
fprintf(warningFid, 'Image feature extraction warnings\n');
fprintf(warningFid, 'Generated: %s\n\n', datestr(now, 31));

bandNames = makeBandNames(cfg.expected_channels);
inputs = cfg.input.files;
inputKeys = fieldnames(inputs);

allObservationRows = cell(numel(inputKeys), 1);
allPatchRows = cell(numel(inputKeys), 1);
metadata = struct();
metadata.created = datestr(now, 31);
metadata.expected_channels = cfg.expected_channels;
metadata.source_files = struct();

fprintf('Starting feature extraction...\n');

for k = 1:numel(inputKeys)
    inputKey = inputKeys{k};
    fileCfg = inputs.(inputKey);
    datasetGroup = string(fileCfg.analysis_label);
    filename = fileCfg.path;

    fprintf('\nReading %s (%s)\n', filename, datasetGroup);
    [patchTablePart, obsTablePart, fileMeta] = extractOneFile( ...
        filename, datasetGroup, cfg, bandNames, warningFid);

    allPatchRows{k} = patchTablePart;
    allObservationRows{k} = obsTablePart;
    metadata.source_files.(inputKey) = fileMeta;
end

patch_table = vertcat(allPatchRows{:}); %#ok<NASGU>
observation_table = vertcat(allObservationRows{:}); %#ok<NASGU>

validateUniqueKeys(patch_table, observation_table, warningFid);

metadata.total_observations = height(observation_table);
metadata.total_patches = height(patch_table);
metadata.band_names = bandNames;
metadata.warning_file = warningFile;
metadata.elapsed_seconds = toc(tStart);

saveCachedTables(cfg, patch_table, observation_table, metadata);
writeCompactSummaries(cfg, observation_table, metadata);

summary = makeSummary(cfg, patch_table, observation_table, metadata, tStart);
printSummary(summary);

end

function ensureOutputDirs(cfg)
dirs = {cfg.paths.cache, cfg.paths.figures, cfg.paths.tables, ...
    cfg.paths.predictions, cfg.paths.logs};
for k = 1:numel(dirs)
    if ~exist(dirs{k}, 'dir')
        mkdir(dirs{k});
    end
end
end

function [patchTable, observationTable, fileMeta] = extractOneFile(filename, datasetGroup, cfg, bandNames, warningFid)
fileMeta = struct();
fileMeta.filename = filename;
fileMeta.dataset_group = datasetGroup;
fileMeta.exists = exist(filename, 'file') == 2;
fileMeta.observations_read = 0;
fileMeta.patches_read = 0;
fileMeta.invalid_observations = strings(0, 1);

if ~fileMeta.exists
    logIssue(warningFid, 'ERROR', filename, NaN, 'File does not exist.');
    patchTable = emptyPatchTable(bandNames);
    observationTable = emptyObservationTable(bandNames);
    error('Input file is missing.');
end

[obsPaths, combinedSize] = resolveObservationReferences(filename, warningFid);
fileMeta.imageRecords_size = combinedSize;
fileMeta.num_observation_references = numel(obsPaths);

if isempty(obsPaths)
    logIssue(warningFid, 'ERROR', filename, NaN, 'No observation references found.');
    patchTable = emptyPatchTable(bandNames);
    observationTable = emptyObservationTable(bandNames);
    return;
end

patchCounts = estimatePatchCounts(filename, obsPaths, warningFid);
totalPatchCapacity = sum(max(patchCounts, 0));
numObsCapacity = numel(obsPaths);

patchCols = initPatchColumns(totalPatchCapacity, numel(bandNames));
obsCols = initObservationColumns(numObsCapacity, numel(bandNames));

patchCursor = 0;
obsCursor = 0;
progressStep = 20;

for obsIdx = 1:numel(obsPaths)
    if obsIdx == 1 || mod(obsIdx, progressStep) == 0 || obsIdx == numel(obsPaths)
        fprintf('  %s: observation %d/%d\n', datasetGroup, obsIdx, numel(obsPaths));
    end

    [record, issues] = readObservation(filename, obsPaths(obsIdx), obsIdx, datasetGroup, cfg);
    if ~isempty(issues)
        for j = 1:numel(issues)
            logIssue(warningFid, 'WARNING', filename, obsIdx, issues(j));
        end
    end

    if ~record.valid
        fileMeta.invalid_observations(end + 1, 1) = string(obsIdx); %#ok<AGROW>
        continue;
    end

    obsCursor = obsCursor + 1;
    obsCols = assignObservationRow(obsCols, obsCursor, record, bandNames);

    nPatches = record.num_patches;
    rows = patchCursor + (1:nPatches);
    patchCols = assignPatchRows(patchCols, rows, record, bandNames);
    patchCursor = patchCursor + nPatches;

    fileMeta.observations_read = fileMeta.observations_read + 1;
    fileMeta.patches_read = fileMeta.patches_read + nPatches;
end

patchCols = trimPatchColumns(patchCols, patchCursor);
obsCols = trimObservationColumns(obsCols, obsCursor);
patchTable = buildPatchTable(patchCols, bandNames);
observationTable = buildObservationTable(obsCols, bandNames);
end

function [obsPaths, combinedSize] = resolveObservationReferences(filename, warningFid)
obsPaths = strings(1, 0);
combinedSize = [];
fileId = [];
datasetId = [];
try
    info = h5info(filename, '/imageRecords');
    combinedSize = info.Dataspace.Size;

    fileId = H5F.open(filename, 'H5F_ACC_RDONLY', 'H5P_DEFAULT');
    datasetId = H5D.open(fileId, '/imageRecords');
    refs = H5D.read(datasetId);
    nRefs = size(refs, 2);
    obsPaths = strings(1, nRefs);
    for idx = 1:nRefs
        obsPaths(idx) = string(H5R.get_name(datasetId, 'H5R_OBJECT', refs(:, idx)));
    end
catch ME
    logIssue(warningFid, 'ERROR', filename, NaN, "Could not resolve HDF5 object references: " + string(ME.message));
end
tryCloseH5(datasetId, "H5D");
tryCloseH5(fileId, "H5F");
end

function patchCounts = estimatePatchCounts(filename, obsPaths, warningFid)
patchCounts = zeros(numel(obsPaths), 1);
for idx = 1:numel(obsPaths)
    try
        totalPatchNumber = h5read(filename, char(obsPaths(idx) + "/total_patch_number"));
        patchCounts(idx) = double(totalPatchNumber(1));
    catch ME
        logIssue(warningFid, 'WARNING', filename, idx, ...
            "Could not read total_patch_number for preallocation: " + string(ME.message));
        try
            info = h5info(filename, char(obsPaths(idx) + "/mean_values"));
            dims = info.Dataspace.Size;
            patchCounts(idx) = max(dims);
        catch
            patchCounts(idx) = 0;
        end
    end
end
end

function [record, issues] = readObservation(filename, obsPath, obsIdx, datasetGroup, cfg)
issues = strings(0, 1);
record = struct();
record.valid = false;
record.source_file = string(filename);
record.dataset_group = datasetGroup;
record.analysis_label = datasetGroup;
record.observation_index = obsIdx;

try
    record.original_class = readScalarString(filename, obsPath + "/class");
    record.domain = readScalarString(filename, obsPath + "/domain");
    record.time = double(readNumericScalar(filename, obsPath + "/time"));
    record.object_number = double(readNumericScalar(filename, obsPath + "/object_number"));
    record.object_name = readScalarString(filename, obsPath + "/object_name");
    record.patch_size = double(readNumericScalar(filename, obsPath + "/patch_size"));
    record.total_patch_number = double(readNumericScalar(filename, obsPath + "/total_patch_number"));
    record.channels = double(h5read(filename, char(obsPath + "/channels"))).';
    locations = double(h5read(filename, char(obsPath + "/locations_yx")));
    meanValues = double(h5read(filename, char(obsPath + "/mean_values")));
catch ME
    issues(end + 1, 1) = "Required field read failed: " + string(ME.message);
    return;
end

[meanValues, orientationOk] = orientMeanValues(meanValues, numel(cfg.expected_channels), record.total_patch_number);
if ~orientationOk
    issues(end + 1, 1) = "mean_values is neither N x C nor C x N after checking dimensions.";
end

if size(meanValues, 2) ~= numel(cfg.expected_channels)
    issues(end + 1, 1) = "mean_values band dimension does not match the configured channel count after orientation correction.";
end
if size(meanValues, 1) ~= record.total_patch_number
    issues(end + 1, 1) = "mean_values patch count does not equal total_patch_number.";
end
if size(locations, 2) ~= 2 && size(locations, 1) == 2
    locations = locations.';
end
if size(locations, 1) ~= record.total_patch_number
    issues(end + 1, 1) = "locations_yx patch count does not equal total_patch_number.";
end
if numel(record.channels) ~= numel(cfg.expected_channels)
    issues(end + 1, 1) = "channel count does not match the configured channel count.";
elseif any(record.channels(:).' ~= cfg.expected_channels)
    issues(end + 1, 1) = "channels do not agree with config.expected_channels.";
end
if ~isnumeric(meanValues) || any(~isfinite(meanValues(:)))
    issues(end + 1, 1) = "mean_values contains nonnumeric or nonfinite values.";
end

if ~isempty(issues)
    return;
end

record.mean_values = meanValues;
record.locations_yx = locations;
record.num_patches = size(meanValues, 1);
record.subject_key = record.object_name;
record.observation_key = record.subject_key + "_" + record.domain + "_time_" + string(record.time);
record.patch_index = (1:record.num_patches).';
record.patch_key = record.observation_key + "_patch_" + string(record.patch_index);
record.valid = true;
end

function [meanValues, ok] = orientMeanValues(meanValues, nBands, totalPatchNumber)
ok = true;
if size(meanValues, 2) == nBands
    return;
elseif size(meanValues, 1) == nBands
    meanValues = meanValues.';
else
    ok = false;
    return;
end

if size(meanValues, 1) ~= totalPatchNumber
    ok = false;
end
end

function value = readNumericScalar(filename, path)
value = h5read(filename, char(path));
value = value(1);
end

function value = readScalarString(filename, path)
raw = h5read(filename, char(path));
if ischar(raw)
    value = string(raw);
elseif isstring(raw)
    value = raw(1);
elseif isa(raw, 'uint16') || isa(raw, 'uint8')
    value = string(char(raw(:).'));
elseif iscell(raw)
    value = string(raw{1});
else
    value = string(raw(1));
end
value = strtrim(value);
end

function patchCols = initPatchColumns(nRows, nBands)
patchCols.source_file = strings(nRows, 1);
patchCols.dataset_group = strings(nRows, 1);
patchCols.original_class = strings(nRows, 1);
patchCols.analysis_label = strings(nRows, 1);
patchCols.domain = strings(nRows, 1);
patchCols.time = NaN(nRows, 1);
patchCols.object_number = NaN(nRows, 1);
patchCols.object_name = strings(nRows, 1);
patchCols.subject_key = strings(nRows, 1);
patchCols.observation_key = strings(nRows, 1);
patchCols.patch_index = NaN(nRows, 1);
patchCols.patch_key = strings(nRows, 1);
patchCols.location_y = NaN(nRows, 1);
patchCols.location_x = NaN(nRows, 1);
patchCols.patch_size = NaN(nRows, 1);
patchCols.bands = NaN(nRows, nBands);
end

function obsCols = initObservationColumns(nRows, nBands)
obsCols.source_file = strings(nRows, 1);
obsCols.dataset_group = strings(nRows, 1);
obsCols.original_class = strings(nRows, 1);
obsCols.analysis_label = strings(nRows, 1);
obsCols.domain = strings(nRows, 1);
obsCols.time = NaN(nRows, 1);
obsCols.object_number = NaN(nRows, 1);
obsCols.object_name = strings(nRows, 1);
obsCols.subject_key = strings(nRows, 1);
obsCols.observation_key = strings(nRows, 1);
obsCols.num_patches = NaN(nRows, 1);
obsCols.patch_size = NaN(nRows, 1);
obsCols.band_mean = NaN(nRows, nBands);
obsCols.band_std = NaN(nRows, nBands);
obsCols.band_median = NaN(nRows, nBands);
end

function patchCols = assignPatchRows(patchCols, rows, record, bandNames) %#ok<INUSD>
n = numel(rows);
patchCols.source_file(rows) = record.source_file;
patchCols.dataset_group(rows) = record.dataset_group;
patchCols.original_class(rows) = record.original_class;
patchCols.analysis_label(rows) = record.analysis_label;
patchCols.domain(rows) = record.domain;
patchCols.time(rows) = record.time;
patchCols.object_number(rows) = record.object_number;
patchCols.object_name(rows) = record.object_name;
patchCols.subject_key(rows) = record.subject_key;
patchCols.observation_key(rows) = record.observation_key;
patchCols.patch_index(rows) = (1:n).';
patchCols.patch_key(rows) = record.patch_key;
patchCols.location_y(rows) = record.locations_yx(:, 1);
patchCols.location_x(rows) = record.locations_yx(:, 2);
patchCols.patch_size(rows) = record.patch_size;
patchCols.bands(rows, :) = record.mean_values;
end

function obsCols = assignObservationRow(obsCols, row, record, bandNames) %#ok<INUSD>
obsCols.source_file(row) = record.source_file;
obsCols.dataset_group(row) = record.dataset_group;
obsCols.original_class(row) = record.original_class;
obsCols.analysis_label(row) = record.analysis_label;
obsCols.domain(row) = record.domain;
obsCols.time(row) = record.time;
obsCols.object_number(row) = record.object_number;
obsCols.object_name(row) = record.object_name;
obsCols.subject_key(row) = record.subject_key;
obsCols.observation_key(row) = record.observation_key;
obsCols.num_patches(row) = record.num_patches;
obsCols.patch_size(row) = record.patch_size;
obsCols.band_mean(row, :) = mean(record.mean_values, 1, 'omitnan');
obsCols.band_std(row, :) = std(record.mean_values, 0, 1, 'omitnan');
obsCols.band_median(row, :) = median(record.mean_values, 1, 'omitnan');
end

function patchCols = trimPatchColumns(patchCols, nRows)
fields = fieldnames(patchCols);
for k = 1:numel(fields)
    name = fields{k};
    patchCols.(name) = patchCols.(name)(1:nRows, :);
end
end

function obsCols = trimObservationColumns(obsCols, nRows)
fields = fieldnames(obsCols);
for k = 1:numel(fields)
    name = fields{k};
    obsCols.(name) = obsCols.(name)(1:nRows, :);
end
end

function tableOut = buildPatchTable(cols, bandNames)
tableOut = table(cols.source_file, cols.dataset_group, cols.original_class, cols.analysis_label, ...
    cols.domain, cols.time, cols.object_number, cols.object_name, cols.subject_key, ...
    cols.observation_key, cols.patch_index, cols.patch_key, cols.location_y, cols.location_x, ...
    cols.patch_size, 'VariableNames', {'source_file', 'dataset_group', 'original_class', ...
    'analysis_label', 'domain', 'time', 'object_number', 'object_name', 'subject_key', ...
    'observation_key', 'patch_index', 'patch_key', 'location_y', 'location_x', 'patch_size'});

bandTable = array2table(cols.bands, 'VariableNames', cellstr(bandNames));
tableOut = [tableOut, bandTable];
end

function tableOut = buildObservationTable(cols, bandNames)
tableOut = table(cols.source_file, cols.dataset_group, cols.original_class, cols.analysis_label, ...
    cols.domain, cols.time, cols.object_number, cols.object_name, cols.subject_key, ...
    cols.observation_key, cols.num_patches, cols.patch_size, 'VariableNames', ...
    {'source_file', 'dataset_group', 'original_class', 'analysis_label', 'domain', ...
    'time', 'object_number', 'object_name', 'subject_key', 'observation_key', 'num_patches', 'patch_size'});

meanNames = "mean_" + bandNames;
stdNames = "std_" + bandNames;
medianNames = "median_" + bandNames;
tableOut = [tableOut, ...
    array2table(cols.band_mean, 'VariableNames', cellstr(meanNames)), ...
    array2table(cols.band_std, 'VariableNames', cellstr(stdNames)), ...
    array2table(cols.band_median, 'VariableNames', cellstr(medianNames))];
end

function tableOut = emptyPatchTable(bandNames)
cols = initPatchColumns(0, numel(bandNames));
tableOut = buildPatchTable(cols, bandNames);
end

function tableOut = emptyObservationTable(bandNames)
cols = initObservationColumns(0, numel(bandNames));
tableOut = buildObservationTable(cols, bandNames);
end

function validateUniqueKeys(patchTable, observationTable, warningFid)
assert(numel(unique(patchTable.patch_key)) == height(patchTable), 'Duplicate patch keys.');
assert(numel(unique(observationTable.observation_key)) == height(observationTable), 'Duplicate observation keys.');
for subject = unique(observationTable.subject_key)'
    assert(numel(unique(observationTable.analysis_label(observationTable.subject_key==subject)))==1, ...
        'A globally identified source has conflicting labels.');
end
if numel(unique(observationTable.observation_key)) ~= height(observationTable)
    logIssue(warningFid, 'ERROR', 'combined', NaN, 'observation_key is not unique in observation table.');
end
if numel(unique(patchTable.patch_key)) ~= height(patchTable)
    logIssue(warningFid, 'ERROR', 'combined', NaN, 'patch_key is not unique in patch table.');
end
end

function saveCachedTables(cfg, patch_table, observation_table, metadata)
patchFile = fullfile(cfg.paths.cache, 'patch_table.mat');
observationFile = fullfile(cfg.paths.cache, 'observation_table.mat');
metadataFile = fullfile(cfg.paths.cache, 'extraction_metadata.mat');
save(patchFile, 'patch_table', '-v7.3');
save(observationFile, 'observation_table', '-v7.3');
save(metadataFile, 'metadata', '-v7.3');
end

function writeCompactSummaries(cfg, observationTable, metadata)
obsCsv = fullfile(cfg.paths.tables, 'observation_table_summary.csv');
metaCsv = fullfile(cfg.paths.tables, 'extraction_summary.csv');
writetable(observationTable, obsCsv);

summaryTable = table( ...
    metadata.total_observations, ...
    metadata.total_patches, ...
    numel(metadata.expected_channels), ...
    metadata.elapsed_seconds, ...
    'VariableNames', {'total_observations', 'total_patches', 'number_of_bands', 'elapsed_seconds'});
writetable(summaryTable, metaCsv);
end

function summary = makeSummary(cfg, patchTable, observationTable, metadata, tStart)
summary = struct();
summary.total_observations = height(observationTable);
summary.total_patches = height(patchTable);
summary.class1_subjects = countSubjects(observationTable, "class1");
summary.class0_subjects = countSubjects(observationTable, "class0");
summary.domains = unique(observationTable.domain);
summary.time_values = unique(observationTable.time);
summary.number_of_bands = numel(metadata.expected_channels);
summary.elapsed_seconds = toc(tStart);
summary.cache_file_sizes = cacheFileSizes(cfg);
end

function n = countSubjects(observationTable, datasetGroup)
rows = observationTable.dataset_group == datasetGroup;
if any(rows)
    n = numel(unique(observationTable.subject_key(rows)));
else
    n = 0;
end
end

function sizes = cacheFileSizes(cfg)
files = ["patch_table.mat", "observation_table.mat", "extraction_metadata.mat"];
sizes = struct();
for k = 1:numel(files)
    filePath = fullfile(cfg.paths.cache, files(k));
    d = dir(filePath);
    fieldName = matlab.lang.makeValidName(erase(files(k), ".mat"));
    if isempty(d)
        sizes.(fieldName) = NaN;
    else
        sizes.(fieldName) = d.bytes;
    end
end
end

function printSummary(summary)
fprintf('\nExtraction complete\n');
fprintf('Total observations: %d\n', summary.total_observations);
fprintf('Total patches: %d\n', summary.total_patches);
fprintf('Class1 subjects: %d\n', summary.class1_subjects);
fprintf('Class0 subjects: %d\n', summary.class0_subjects);
fprintf('Domains: %s\n', strjoin(string(summary.domains), ', '));
fprintf('TIME values: %s\n', strjoin(string(summary.time_values.'), ', '));
fprintf('Number of bands: %d\n', summary.number_of_bands);
fprintf('Elapsed time: %.2f seconds\n', summary.elapsed_seconds);
fprintf('Cache file sizes:\n');
fields = fieldnames(summary.cache_file_sizes);
for k = 1:numel(fields)
    bytes = summary.cache_file_sizes.(fields{k});
    fprintf('  %s: %.3f MB\n', fields{k}, bytes / 1024^2);
end
end

function bandNames = makeBandNames(channels)
bandNames = "band_" + string(channels);
end

function logIssue(fid, level, filename, obsIdx, message)
if isnan(obsIdx)
    fprintf(fid, '%s | file=%s | %s\n', level, string(filename), string(message));
else
    fprintf(fid, '%s | file=%s | observation=%d | %s\n', level, string(filename), obsIdx, string(message));
end
end

function tryCloseH5(id, kind)
try
    if isempty(id)
        return;
    end
    if kind == "H5D"
        H5D.close(id);
    elseif kind == "H5F"
        H5F.close(id);
    end
catch
end
end
