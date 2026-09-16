function varargout = load_cached_data(whichData)
% Adapted from the original analysis implementation; identifiers and I/O generalized.
%LOAD_CACHED_DATA Load cached image feature tables.

if nargin < 1 || strlength(string(whichData)) == 0
    whichData = "both";
end
whichData = lower(string(whichData));

cfg = config();
patchFile = fullfile(cfg.paths.cache, 'patch_table.mat');
observationFile = fullfile(cfg.paths.cache, 'observation_table.mat');
metadataFile = fullfile(cfg.paths.cache, 'extraction_metadata.mat');

switch whichData
    case "observation"
        assertCacheFile(observationFile, 'observation table');
        observation_table = loadVariable(observationFile, 'observation_table');
        validateObservationTable(observation_table);
        metadata = loadMetadataIfAvailable(metadataFile);
        reportCacheStatus(metadata, [], observation_table);
        varargout = {observation_table};

    case "patch"
        assertCacheFile(patchFile, 'patch table');
        patch_table = loadVariable(patchFile, 'patch_table');
        validatePatchTable(patch_table);
        metadata = loadMetadataIfAvailable(metadataFile);
        reportCacheStatus(metadata, patch_table, []);
        varargout = {patch_table};

    case "both"
        assertCacheFile(patchFile, 'patch table');
        assertCacheFile(observationFile, 'observation table');
        patch_table = loadVariable(patchFile, 'patch_table');
        observation_table = loadVariable(observationFile, 'observation_table');
        validatePatchTable(patch_table);
        validateObservationTable(observation_table);
        metadata = loadMetadataIfAvailable(metadataFile);
        reportCacheStatus(metadata, patch_table, observation_table);
        varargout = {patch_table, observation_table};

    otherwise
        error('Unknown cache request "%s". Use "observation", "patch", or "both".', whichData);
end

end

function value = loadVariable(filename, variableName)
try
    m = matfile(filename);
    vars = who(m);
    if ~any(strcmp(vars, variableName))
        error('VariableMissing:CacheVariable', ...
            'Cache file %s does not contain variable "%s". Run extract_feature_tables.m again.', ...
            filename, variableName);
    end
    value = m.(variableName);
catch ME
    if strcmp(ME.identifier, 'VariableMissing:CacheVariable')
        rethrow(ME);
    end
    error('Could not load "%s" from %s: %s', variableName, filename, ME.message);
end
end

function assertCacheFile(filename, description)
if exist(filename, 'file') ~= 2
    error(['Missing cached %s file: %s\n' ...
        'Run extract_feature_tables.m first to create the lightweight cache.'], ...
        description, filename);
end
end

function metadata = loadMetadataIfAvailable(metadataFile)
metadata = struct();
if exist(metadataFile, 'file') ~= 2
    return;
end
try
    m = matfile(metadataFile);
    vars = who(m);
    if any(strcmp(vars, 'metadata'))
        metadata = m.metadata;
    end
catch
    metadata = struct();
end
end

function validatePatchTable(tbl)
required = ["source_file", "dataset_group", "original_class", "analysis_label", ...
    "domain", "time", "object_number", "object_name", "subject_key", ...
    "observation_key", "patch_index", "patch_key", "location_y", "location_x", ...
    "patch_size", "band_" + string(config().expected_channels)];
validateTableColumns(tbl, required, 'patch_table');
end

function validateObservationTable(tbl)
required = ["source_file", "dataset_group", "original_class", "analysis_label", ...
    "domain", "time", "object_number", "object_name", "subject_key", ...
    "observation_key", "num_patches", "patch_size", ...
    "mean_band_" + string(config().expected_channels), ...
    "std_band_" + string(config().expected_channels), ...
    "median_band_" + string(config().expected_channels)];
validateTableColumns(tbl, required, 'observation_table');
end

function validateTableColumns(tbl, required, tableName)
if ~istable(tbl)
    error('Cached variable "%s" is not a table. Run extract_feature_tables.m again.', tableName);
end
missing = setdiff(required, string(tbl.Properties.VariableNames), 'stable');
if ~isempty(missing)
    error('Cached %s is missing required columns: %s. Run extract_feature_tables.m again.', ...
        tableName, strjoin(missing, ', '));
end
end

function reportCacheStatus(metadata, patchTable, observationTable)
fprintf('Cached image data loaded.\n');
if isfield(metadata, 'created')
    fprintf('Cache creation date: %s\n', string(metadata.created));
else
    fprintf('Cache creation date: unavailable\n');
end

if ~isempty(patchTable)
    fprintf('Patch table rows: %d\n', height(patchTable));
end
if ~isempty(observationTable)
    fprintf('Observation table rows: %d\n', height(observationTable));
end
if isfield(metadata, 'total_patches') && isempty(patchTable)
    fprintf('Patch table rows from metadata: %d\n', metadata.total_patches);
end
if isfield(metadata, 'total_observations') && isempty(observationTable)
    fprintf('Observation table rows from metadata: %d\n', metadata.total_observations);
end
end
