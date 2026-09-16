function import_feature_table(X, meta)
%IMPORT_FEATURE_TABLE Cache private patch features and observation summaries.
% X is N-by-C. meta requires analysis_label, subject_key, observation_key,
% domain, time. Each subject is one independent source, globally identified.
cfg = config();
validateattributes(X, {'numeric'}, {'2d','nonempty','finite','real'});
required = ["analysis_label","subject_key","observation_key","domain","time"];
assert(istable(meta) && height(meta) == size(X,1), 'Metadata must have one row per patch.');
assert(all(ismember(required, string(meta.Properties.VariableNames))), 'Required metadata columns are missing.');
assert(size(X,2) == numel(cfg.expected_channels), 'Feature count differs from configuration.');
for name = required(1:4)
    meta.(name) = string(meta.(name));
    assert(all(~ismissing(meta.(name)) & strlength(strtrim(meta.(name))) > 0), 'Metadata strings cannot be empty.');
end
validateattributes(meta.time, {'numeric'}, {'column','finite','nonnegative'});
assert(all(ismember(meta.analysis_label, ["class0","class1"])), 'Use generic class0 and class1 labels.');
subjects = unique(meta.subject_key);
for i = 1:numel(subjects)
    assert(numel(unique(meta.analysis_label(meta.subject_key == subjects(i)))) == 1, 'A source has conflicting labels.');
end
N = height(meta);
patch_table = meta(:, cellstr(required));
patch_table.source_file = repmat("private_input", N, 1);
patch_table.dataset_group = meta.analysis_label;
patch_table.original_class = meta.analysis_label;
[~, patch_table.object_number] = ismember(meta.subject_key, subjects);
patch_table.object_name = meta.subject_key;
patch_table.patch_size = zeros(N,1); % Unknown for precomputed features.
patch_table.location_y = NaN(N,1);
patch_table.location_x = NaN(N,1);
patch_table.patch_index = zeros(N,1);
patch_table.patch_key = strings(N,1);
bands = "band_" + string(cfg.expected_channels);
patch_table = [patch_table array2table(double(X), 'VariableNames', cellstr(bands))];
keys = unique(meta.observation_key, 'stable');
rows = cell(numel(keys),1);
for i = 1:numel(keys)
    idx = find(meta.observation_key == keys(i));
    for name = ["analysis_label","subject_key","domain","time"]
        assert(numel(unique(meta.(name)(idx))) == 1, 'Observation metadata must be consistent.');
    end
    patch_table.patch_index(idx) = (1:numel(idx))';
    patch_table.patch_key(idx) = keys(i) + "_patch_" + string((1:numel(idx))');
    row = patch_table(idx(1), {'source_file','dataset_group','original_class','analysis_label', ...
        'domain','time','object_number','object_name','subject_key','observation_key','patch_size'});
    row.num_patches = numel(idx);
    row = [row array2table(mean(X(idx,:),1), 'VariableNames', cellstr("mean_" + bands)), ...
        array2table(std(X(idx,:),0,1), 'VariableNames', cellstr("std_" + bands)), ...
        array2table(median(X(idx,:),1), 'VariableNames', cellstr("median_" + bands))];
    rows{i} = row;
end
observation_table = vertcat(rows{:});
metadata = struct('created', datestr(now,31), 'total_patches',N, ...
    'total_observations',height(observation_table), 'expected_channels',cfg.expected_channels);
if ~isfolder(cfg.paths.cache), mkdir(cfg.paths.cache); end
save(fullfile(cfg.paths.cache,'patch_table.mat'), 'patch_table', '-v7.3');
save(fullfile(cfg.paths.cache,'observation_table.mat'), 'observation_table', '-v7.3');
save(fullfile(cfg.paths.cache,'extraction_metadata.mat'), 'metadata', '-v7.3');
end
