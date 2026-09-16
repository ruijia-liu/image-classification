function report = inspect_mat_files()
% Adapted from the original analysis implementation; identifiers and I/O generalized.
%INSPECT_MAT_FILES Inspect image MATLAB v7.3 MAT files without extraction.

cfg = config();
logFile = fullfile(cfg.paths.logs, 'mat_file_inspection.txt');
if ~exist(cfg.paths.logs, 'dir')
    mkdir(cfg.paths.logs);
end

fid = fopen(logFile, 'w');
if fid < 0
    error('Could not open inspection report for writing: %s', logFile);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

rng(cfg.RANDOM_SEED);
fprintf(fid, 'Image multispectral MAT file inspection\n');
fprintf(fid, 'Generated: %s\n', datestr(now, 31));
fprintf(fid, 'Report path: %s\n\n', logFile);

keys = fieldnames(cfg.input.files);
report = struct();

for k = 1:numel(keys)
    key = keys{k};
    fileCfg = cfg.input.files.(key);
    report.(key) = inspectOneMatFile(fileCfg.path, fileCfg.analysis_label, cfg, fid);
end

fprintf(fid, '\nCross-file checks\n');
fprintf(fid, '=================\n');
compareFieldStructure(report, keys, fid);
compareChannels(report, keys, fid);
checkCrossFileObjectOverlap(report, keys, fid);

fprintf('Inspection report saved to: %s\n', logFile);

end

function out = inspectOneMatFile(filename, analysisLabel, cfg, fid)
out = struct();
out.filename = filename;
out.analysis_label = string(analysisLabel);
out.exists = exist(filename, 'file') == 2;
out.top_level_variables = struct([]);
out.imageRecords = struct('available', false, 'class', "", 'size', []);
out.imageRecords_storage = "unknown";
out.num_observations = NaN;
out.field_names = string.empty(1, 0);
out.channel_signatures = strings(0, 1);
out.object_records = table();

fprintf(fid, '\nFile: %s\n', filename);
fprintf(fid, 'Analysis label assigned by config: %s\n', string(analysisLabel));
fprintf(fid, '%s\n', repmat('=', 1, 72));

if ~out.exists
    warn(fid, 'File does not exist: %s', filename);
    return;
end

d = dir(filename);
fprintf(fid, 'File size: %.3f MB (%d bytes)\n', d.bytes / 1024^2, d.bytes);

out.hdf5_status = inspectHdf5Status(filename, fid);
out.top_level_variables = inspectTopLevelVariables(filename, fid);
out.imageRecords = inspectImageRecordsVariable(out.top_level_variables, fid);

h5root = [];
try
    h5root = h5info(filename);
catch ME
    warn(fid, 'h5info failed: %s', ME.message);
end
out.imageRecords_storage = inspectH5ImageRecords(filename, h5root, fid);
out.field_names = getH5FieldNames(h5root);

mf = [];
try
    mf = matfile(filename);
    fprintf(fid, 'matfile status: available\n');
catch ME
    warn(fid, 'matfile could not open the file: %s', ME.message);
end

if isempty(mf)
    warn(fid, 'Skipping record-level checks because matfile is unavailable.');
    return;
end

[out.num_observations, cpSize] = getObservationCount(mf, out.imageRecords.size, fid);
fprintf(fid, 'imageRecords matfile size: %s\n', mat2str(cpSize));
fprintf(fid, 'Number of observation records: %s\n', scalarString(out.num_observations));

if isnan(out.num_observations) || out.num_observations < 1
    warn(fid, 'No observation records detected.');
    return;
end

sampleIdx = unique([1, min(2, out.num_observations), min(3, out.num_observations), ...
    max(1, floor(out.num_observations / 2)), out.num_observations]);
fprintf(fid, 'Sampled observation indices: %s\n', mat2str(sampleIdx));

out.h5_observation_paths = getH5ObservationPaths(filename, out.num_observations, fid);
if isempty(out.field_names) && ~isempty(out.h5_observation_paths)
    out.field_names = getObservationFieldNames(filename, out.h5_observation_paths(1), fid);
end

[out.field_names, out.field_summary] = inspectFieldTypesAndDimensions(filename, mf, h5root, ...
    out.h5_observation_paths, out.field_names, sampleIdx, fid);
out.examples = inspectExampleValues(filename, mf, h5root, out.h5_observation_paths, sampleIdx, cfg, fid);
out.channel_signatures = inspectChannelConsistency(filename, mf, out.h5_observation_paths, ...
    out.num_observations, cfg, fid);
out.object_records = collectObjectRecords(filename, mf, out.h5_observation_paths, out.num_observations, analysisLabel, fid);
checkObjectRepeatsAcrossTime(out.object_records, analysisLabel, fid);

end

function hdf5Status = inspectHdf5Status(filename, fid)
hdf5Status = struct('is_hdf5', false, 'matlab_version', "unknown");
try
    fileId = H5F.open(filename, 'H5F_ACC_RDONLY', 'H5P_DEFAULT');
    H5F.close(fileId);
    hdf5Status.is_hdf5 = true;
    hdf5Status.matlab_version = "v7.3/HDF5-compatible";
    fprintf(fid, 'MATLAB version/HDF5 status: HDF5 readable, likely MATLAB v7.3\n');
catch ME
    fprintf(fid, 'MATLAB version/HDF5 status: not readable as HDF5 by H5F.open\n');
    warn(fid, 'HDF5 detail: %s', ME.message);
end
end

function vars = inspectTopLevelVariables(filename, fid)
vars = struct([]);
fprintf(fid, '\nTop-level variables\n');
fprintf(fid, '-------------------\n');
try
    vars = whos('-file', filename);
    if isempty(vars)
        fprintf(fid, '(none)\n');
    end
    for k = 1:numel(vars)
        fprintf(fid, '%s | class=%s | size=%s | bytes=%d\n', ...
            vars(k).name, vars(k).class, mat2str(vars(k).size), vars(k).bytes);
    end
catch ME
    warn(fid, 'whos("-file", filename) failed: %s', ME.message);
end
end

function cp = inspectImageRecordsVariable(vars, fid)
cp = struct('available', false, 'class', "", 'size', []);
fprintf(fid, '\nimageRecords variable\n');
fprintf(fid, '------------------------\n');
if isempty(vars)
    warn(fid, 'No top-level variable metadata available.');
    return;
end
idx = find(strcmp({vars.name}, 'imageRecords'), 1);
if isempty(idx)
    warn(fid, 'imageRecords was not found as a top-level variable.');
    return;
end
cp.available = true;
cp.class = string(vars(idx).class);
cp.size = vars(idx).size;
fprintf(fid, 'Size: %s\n', mat2str(cp.size));
fprintf(fid, 'Type: %s\n', cp.class);
end

function storage = inspectH5ImageRecords(filename, h5root, fid)
storage = "unknown";
fprintf(fid, '\nimageRecords HDF5 representation\n');
fprintf(fid, '------------------------------------\n');
if isempty(h5root)
    warn(fid, 'Cannot inspect HDF5 representation because h5info is unavailable.');
    return;
end

try
    groupNames = string({h5root.Groups.Name});
    datasetNames = string({h5root.Datasets.Name});
    if any(groupNames == "/imageRecords")
        cp = h5info(filename, '/imageRecords');
        storage = classifyH5Group(cp);
        fprintf(fid, 'Node: /imageRecords group\n');
        fprintf(fid, 'Storage classification: %s\n', storage);
        fprintf(fid, 'Groups: %d | Datasets: %d | Attributes: %d\n', ...
            numel(cp.Groups), numel(cp.Datasets), numel(cp.Attributes));
        printFirstH5Children(cp, fid);
    elseif any(datasetNames == "imageRecords")
        ds = h5info(filename, '/imageRecords');
        storage = classifyH5Dataset(ds);
        fprintf(fid, 'Node: /imageRecords dataset\n');
        fprintf(fid, 'Storage classification: %s\n', storage);
        fprintf(fid, 'Dataset size: %s | datatype: %s\n', mat2str(ds.Dataspace.Size), ds.Datatype.Class);
    else
        warn(fid, 'No /imageRecords node found at HDF5 file root.');
    end
catch ME
    warn(fid, 'Could not inspect /imageRecords with h5info: %s', ME.message);
end
end

function storage = classifyH5Group(groupInfo)
datasetNames = string({groupInfo.Datasets.Name});
groupNames = string({groupInfo.Groups.Name});
if any(datasetNames == "#refs#") || any(contains(groupNames, "/#refs#"))
    storage = "HDF5 object references";
elseif any(datasetNames == "MATLAB_fields") || any(contains(datasetNames, "field"))
    storage = "MATLAB struct array";
elseif any(endsWith(groupNames, "/cropped_patches")) || any(datasetNames == "cropped_patches")
    storage = "nested MATLAB/HDF5 representation with fields";
elseif ~isempty(groupInfo.Groups)
    storage = "another nested representation";
elseif ~isempty(groupInfo.Datasets)
    storage = "HDF5 dataset-backed representation";
else
    storage = "unknown";
end
end

function storage = classifyH5Dataset(datasetInfo)
if contains(lower(string(datasetInfo.Datatype.Class)), "reference")
    storage = "HDF5 object references";
else
    storage = "HDF5 dataset-backed representation";
end
end

function printFirstH5Children(groupInfo, fid)
for k = 1:min(12, numel(groupInfo.Datasets))
    ds = groupInfo.Datasets(k);
    fprintf(fid, 'Dataset: %s | size=%s | datatype=%s\n', ...
        ds.Name, mat2str(ds.Dataspace.Size), ds.Datatype.Class);
end
for k = 1:min(12, numel(groupInfo.Groups))
    fprintf(fid, 'Group: %s\n', groupInfo.Groups(k).Name);
end
end

function names = getH5FieldNames(h5root)
names = string.empty(1, 0);
if isempty(h5root)
    return;
end
cp = getRootGroup(h5root, '/imageRecords');
if isempty(cp)
    return;
end
datasetNames = string({cp.Datasets.Name});
groupNames = string({cp.Groups.Name});
datasetNames = datasetNames(datasetNames ~= "#refs#" & datasetNames ~= "MATLAB_fields");
groupNames = erase(groupNames, "/imageRecords/");
groupNames = groupNames(groupNames ~= "#refs#");
names = unique([datasetNames, groupNames], 'stable');
end

function paths = getH5ObservationPaths(filename, nObs, fid)
paths = strings(1, 0);
try
    fileId = H5F.open(filename, 'H5F_ACC_RDONLY', 'H5P_DEFAULT');
    datasetId = H5D.open(fileId, '/imageRecords');
    refs = H5D.read(datasetId);
    nRefs = size(refs, 2);
    nUse = min(nObs, nRefs);
    paths = strings(1, nUse);
    for idx = 1:nUse
        ref = refs(:, idx);
        paths(idx) = string(H5R.get_name(datasetId, 'H5R_OBJECT', ref));
    end
    H5D.close(datasetId);
    H5F.close(fileId);
    fprintf(fid, 'Resolved HDF5 object references: %d\n', numel(paths));
    if ~isempty(paths)
        fprintf(fid, 'First observation reference path: %s\n', paths(1));
    end
catch ME
    warn(fid, 'Could not resolve HDF5 object references: %s', ME.message);
    tryCloseH5('datasetId', 'H5D');
    tryCloseH5('fileId', 'H5F');
end
end

function names = getObservationFieldNames(filename, obsPath, fid)
names = string.empty(1, 0);
try
    info = h5info(filename, char(obsPath));
    datasetNames = string.empty(1, 0);
    groupNames = string.empty(1, 0);
    if isfield(info, 'Datasets') && isstruct(info.Datasets)
        for k = 1:numel(info.Datasets)
            datasetNames(end + 1) = string(info.Datasets(k).Name); %#ok<AGROW>
        end
    end
    if isfield(info, 'Groups') && isstruct(info.Groups)
        for k = 1:numel(info.Groups)
            groupNames(end + 1) = string(info.Groups(k).Name); %#ok<AGROW>
        end
    end
    groupNames = erase(groupNames, obsPath + "/");
    names = unique([datasetNames, groupNames], 'stable');
    fprintf(fid, 'Field names from first observation reference: %s\n', joinOrEmpty(names));
catch ME
    warn(fid, 'Could not inspect first observation reference %s: %s', obsPath, ME.message);
end
end

function tryCloseH5(varName, kind)
try
    value = evalin('caller', varName);
    if strcmp(kind, 'H5D')
        H5D.close(value);
    elseif strcmp(kind, 'H5F')
        H5F.close(value);
    end
catch
end
end

function groupInfo = getRootGroup(h5root, groupName)
groupInfo = [];
for k = 1:numel(h5root.Groups)
    if string(h5root.Groups(k).Name) == string(groupName)
        groupInfo = h5root.Groups(k);
        return;
    end
end
end

function [nObs, cpSize] = getObservationCount(mf, fallbackSize, fid)
nObs = NaN;
cpSize = fallbackSize;
try
    props = whos(mf, 'imageRecords');
    if ~isempty(props)
        cpSize = props.size;
    end
catch ME
    warn(fid, 'whos(matfile, "imageRecords") failed: %s', ME.message);
end
if ~isempty(cpSize)
    nObs = max(cpSize);
end
end

function [fieldNames, summary] = inspectFieldTypesAndDimensions(filename, mf, h5root, h5Paths, fieldNames, sampleIdx, fid)
summary = struct();
fprintf(fid, '\nField names, data types, and dimensions\n');
fprintf(fid, '---------------------------------------\n');

if isempty(fieldNames)
    fieldNames = inferFieldNamesFromSample(mf, sampleIdx, fid);
else
    fieldNames = reshape(string(fieldNames), 1, []);
    fprintf(fid, 'Field names from HDF5 metadata: %s\n', strjoin(fieldNames, ', '));
end
fieldNames = reshape(string(fieldNames), 1, []);

if isempty(fieldNames)
    warn(fid, 'No field names could be detected.');
    return;
end

if ~any(fieldNames == "cropped_patches")
    cropMeta = croppedPatchesMetadata(filename, h5root, h5Paths);
    if cropMeta.available
        fieldNames = [fieldNames, "cropped_patches"];
    end
end

for f = 1:numel(fieldNames)
    name = fieldNames(f);
    classes = strings(0, 1);
    dims = strings(0, 1);

    if name == "cropped_patches"
        cropMeta = croppedPatchesMetadata(filename, h5root, h5Paths);
        if cropMeta.available
            classes = cropMeta.class;
            dims = cropMeta.dimensions;
        else
            warn(fid, 'cropped_patches metadata unavailable without loading the field.');
        end
    else
        for idx = sampleIdx
            [ok, value] = readRecordField(filename, mf, h5Paths, idx, name, fid);
            if ok
                classes(end + 1, 1) = string(class(value)); %#ok<AGROW>
                dims(end + 1, 1) = string(mat2str(size(value))); %#ok<AGROW>
            end
        end
    end

    classes = unique(classes, 'stable');
    dims = unique(dims, 'stable');
    summary.(matlab.lang.makeValidName(name)) = struct('classes', classes, 'dimensions', dims);
    fprintf(fid, '%s | type(s): %s | dimension(s): %s\n', name, joinOrEmpty(classes), joinOrEmpty(dims));
end
end

function fieldNames = inferFieldNamesFromSample(mf, sampleIdx, fid)
fieldNames = string.empty(1, 0);
for idx = sampleIdx
    try
        [okRec, rec] = readObservationRecord(mf, idx, fid);
        if okRec && isstruct(rec)
            names = string(fieldnames(rec));
            names = names(names ~= "cropped_patches");
            fieldNames = union(fieldNames, names, 'stable');
        elseif okRec
            warn(fid, 'imageRecords record %d loaded as %s rather than struct.', idx, class(rec));
        else
            warn(fid, 'Could not infer fields from imageRecords record %d.', idx);
        end
    catch ME
        warn(fid, 'Could not infer fields from imageRecords(%d): %s', idx, ME.message);
    end
end
end

function examples = inspectExampleValues(filename, mf, h5root, h5Paths, sampleIdx, cfg, fid)
examples = struct();
fprintf(fid, '\nExample values\n');
fprintf(fid, '--------------\n');

valueFields = ["class", "time", "domain", "object_name", "object_number", ...
    "patch_size", "total_patch_number", "channels"];
for f = 1:numel(valueFields)
    name = valueFields(f);
    values = strings(0, 1);
    for idx = sampleIdx
        [ok, value] = readRecordField(filename, mf, h5Paths, idx, name, fid);
        if ok
            values(end + 1, 1) = summarizeValue(value); %#ok<AGROW>
        end
    end
    values = unique(values, 'stable');
    examples.(matlab.lang.makeValidName(name)) = values;
    fprintf(fid, '%s examples: %s\n', name, joinOrEmpty(values(1:min(numel(values), 5))));
end

dimensionFields = ["mean_values", "locations_yx"];
for f = 1:numel(dimensionFields)
    name = dimensionFields(f);
    dims = strings(0, 1);
    for idx = sampleIdx
        [ok, value] = readRecordField(filename, mf, h5Paths, idx, name, fid);
        if ok
            dims(end + 1, 1) = string(mat2str(size(value))); %#ok<AGROW>
        end
    end
    dims = unique(dims, 'stable');
    examples.(matlab.lang.makeValidName(name + "_dimensions")) = dims;
    fprintf(fid, '%s dimensions: %s\n', name, joinOrEmpty(dims));
end

    cropMeta = croppedPatchesMetadata(filename, h5root, h5Paths);
if cropMeta.available
    fprintf(fid, 'cropped_patches reference count or dimensions: %s\n', cropMeta.dimensions);
    fprintf(fid, 'cropped_patches metadata type: %s\n', cropMeta.class);
else
    warn(fid, 'cropped_patches reference count or dimensions unavailable without loading the field.');
end
examples.cropped_patches_dimensions = cropMeta.dimensions;

fprintf(fid, 'Expected channels from config: %s\n', mat2str(cfg.expected_channels));
end

function signatures = inspectChannelConsistency(filename, mf, h5Paths, nObs, cfg, fid)
fprintf(fid, '\nChannel consistency within file\n');
fprintf(fid, '----------------------------------\n');
signatures = strings(0, 1);
idxToCheck = unique(round(linspace(1, nObs, min(nObs, 200))));

for idx = idxToCheck
    [ok, value] = readRecordField(filename, mf, h5Paths, idx, "channels", fid);
    if ok
        signatures(end + 1, 1) = numericSignature(value); %#ok<AGROW>
    end
end

signatures = unique(signatures, 'stable');
fprintf(fid, 'Checked observations: %d of %d\n', numel(idxToCheck), nObs);
fprintf(fid, 'Unique channel signatures: %d\n', numel(signatures));
for k = 1:min(5, numel(signatures))
    fprintf(fid, 'Signature %d: %s\n', k, signatures(k));
end

expected = numericSignature(cfg.expected_channels);
if isempty(signatures)
    warn(fid, 'No channels could be checked.');
elseif numel(signatures) > 1
    warn(fid, 'Channels vary across checked observations.');
elseif signatures(1) ~= expected
    warn(fid, 'Channels do not exactly match cfg.expected_channels.');
else
    fprintf(fid, 'Channels match cfg.expected_channels in checked observations.\n');
end
end

function records = collectObjectRecords(filename, mf, h5Paths, nObs, analysisLabel, fid)
fprintf(fid, '\nObject identifiers for repetition/overlap checks\n');
fprintf(fid, '----------------------------------------------\n');
idxToCheck = unique(round(linspace(1, nObs, min(nObs, 500))));
analysis_label = strings(0, 1);
domain = strings(0, 1);
time = strings(0, 1);
object_number = strings(0, 1);

for idx = idxToCheck
    [okE, e] = readRecordField(filename, mf, h5Paths, idx, "domain", fid);
    [okD, d] = readRecordField(filename, mf, h5Paths, idx, "time", fid);
    [okL, l] = readRecordField(filename, mf, h5Paths, idx, "object_number", fid);
    if okE && okD && okL
        analysis_label(end + 1, 1) = string(analysisLabel); %#ok<AGROW>
        domain(end + 1, 1) = scalarToString(e); %#ok<AGROW>
        time(end + 1, 1) = scalarToString(d); %#ok<AGROW>
        object_number(end + 1, 1) = scalarToString(l); %#ok<AGROW>
    end
end

records = table(analysis_label, domain, time, object_number);
fprintf(fid, 'Identifier records checked: %d of %d observations\n', height(records), nObs);
end

function checkObjectRepeatsAcrossTime(records, analysisLabel, fid)
if height(records) == 0
    warn(fid, 'Cannot check object_number repeats across TIME for %s.', string(analysisLabel));
    return;
end

keys = records.domain + "|" + records.object_number;
uniqueKeys = unique(keys);
repeats = strings(0, 1);
for k = 1:numel(uniqueKeys)
    rows = keys == uniqueKeys(k);
    times = unique(records.time(rows));
    if numel(times) > 1
        repeats(end + 1, 1) = uniqueKeys(k) + " across TIME " + strjoin(times, ', '); %#ok<AGROW>
    end
end

if isempty(repeats)
    fprintf(fid, 'Object repeat check for %s: no domain + object_number repeats across TIME in checked records.\n', string(analysisLabel));
else
    warn(fid, 'object_number repeats across TIME within domain for %s.', string(analysisLabel));
    fprintf(fid, 'First repeated keys: %s\n', strjoin(repeats(1:min(numel(repeats), 20)), ' | '));
end
end

function compareFieldStructure(report, keys, fid)
if numel(keys) < 2
    return;
end
a = report.(keys{1}).field_names;
b = report.(keys{2}).field_names;
if isempty(a) || isempty(b)
    warn(fid, 'Cannot compare field structure because one or both field lists are empty.');
    return;
end
onlyA = setdiff(a, b);
onlyB = setdiff(b, a);
if isempty(onlyA) && isempty(onlyB)
    fprintf(fid, 'Field structure: class1 and class0 files match in detected fields.\n');
else
    warn(fid, 'Field structures differ between class1 and class0 files.');
    fprintf(fid, 'Only in %s: %s\n', keys{1}, joinOrEmpty(onlyA));
    fprintf(fid, 'Only in %s: %s\n', keys{2}, joinOrEmpty(onlyB));
end
end

function compareChannels(report, keys, fid)
if numel(keys) < 2
    return;
end
a = report.(keys{1}).channel_signatures;
b = report.(keys{2}).channel_signatures;
if isempty(a) || isempty(b)
    warn(fid, 'Cannot compare channels across files because one or both files lack channel signatures.');
    return;
end
if isequal(sort(a), sort(b))
    fprintf(fid, 'Channels across files: checked signatures match.\n');
else
    warn(fid, 'Channel signatures differ across files.');
    fprintf(fid, '%s signatures: %s\n', keys{1}, joinOrEmpty(a));
    fprintf(fid, '%s signatures: %s\n', keys{2}, joinOrEmpty(b));
end
end

function checkCrossFileObjectOverlap(report, keys, fid)
if numel(keys) < 2
    return;
end
a = report.(keys{1}).object_records;
b = report.(keys{2}).object_records;
if height(a) == 0 || height(b) == 0
    warn(fid, 'Cannot check domain + object_number overlap between files.');
    return;
end
aKey = a.domain + "|" + a.object_number;
bKey = b.domain + "|" + b.object_number;
overlap = intersect(unique(aKey), unique(bKey));
if isempty(overlap)
    fprintf(fid, 'Domain + object_number overlap between class1 and class0: none in checked records.\n');
else
    warn(fid, 'Domain + object_number values overlap between class1 and class0 checked records.');
    fprintf(fid, 'First overlapping keys: %s\n', strjoin(overlap(1:min(numel(overlap), 20)), ', '));
end
end

function [ok, value] = readRecordField(filename, mf, h5Paths, idx, fieldName, fid)
ok = false;
value = [];
if fieldName == "cropped_patches"
    warn(fid, 'Refusing to load cropped_patches at record %d; using HDF5 metadata only.', idx);
    return;
end
if ~isempty(h5Paths) && idx <= numel(h5Paths) && strlength(h5Paths(idx)) > 0
    fieldPath = h5Paths(idx) + "/" + fieldName;
    try
        value = h5read(filename, char(fieldPath));
        value = decodeMatlabH5Value(value);
        ok = true;
        return;
    catch ME
        warn(fid, 'Could not read HDF5 field %s at record %d: %s', fieldName, idx, ME.message);
    end
end
try
    [okRec, rec] = readObservationRecord(mf, idx, fid);
    if ~okRec
        return;
    end
    if isstruct(rec) && isfield(rec, fieldName)
        value = rec.(fieldName);
        ok = true;
    else
        warn(fid, 'Optional field unavailable at record %d: %s', idx, fieldName);
    end
catch ME
    warn(fid, 'Could not read field %s at record %d: %s', fieldName, idx, ME.message);
end
end

function value = decodeMatlabH5Value(value)
if isa(value, 'uint16') && isvector(value)
    value = char(value(:).');
elseif isa(value, 'uint8') && isvector(value)
    printable = all(value(:) >= 9 & value(:) <= 126);
    if printable
        value = char(value(:).');
    end
end
end

function [ok, rec] = readObservationRecord(mf, idx, fid)
ok = false;
rec = [];
try
    cp = mf.imageRecords(1, idx);
    if iscell(cp)
        if isempty(cp)
            warn(fid, 'imageRecords(1,%d) is an empty cell.', idx);
            return;
        end
        rec = cp{1};
    else
        rec = cp;
    end
    ok = true;
catch ME
    warn(fid, 'Could not read imageRecords(1,%d): %s', idx, ME.message);
end
end

function meta = croppedPatchesMetadata(filename, h5root, h5Paths)
meta = struct('available', false, 'class', "", 'dimensions', "");
if ~isempty(h5Paths) && strlength(h5Paths(1)) > 0
    cropPath = h5Paths(1) + "/cropped_patches";
    try
        info = h5info(filename, char(cropPath));
        meta.available = true;
        if isfield(info, 'Datatype')
            meta.class = "HDF5 dataset " + string(info.Datatype.Class);
            meta.dimensions = string(mat2str(info.Dataspace.Size));
        else
            meta.class = "HDF5 group";
            meta.dimensions = "datasets=" + string(numel(info.Datasets)) + ", groups=" + string(numel(info.Groups));
        end
        return;
    catch
    end
end
if isempty(h5root)
    return;
end
cp = getRootGroup(h5root, '/imageRecords');
if isempty(cp)
    return;
end

datasetNames = string({cp.Datasets.Name});
idx = find(datasetNames == "cropped_patches", 1);
if ~isempty(idx)
    ds = cp.Datasets(idx);
    meta.available = true;
    meta.class = "HDF5 dataset " + string(ds.Datatype.Class);
    meta.dimensions = string(mat2str(ds.Dataspace.Size));
    return;
end

groupNames = string({cp.Groups.Name});
idx = find(endsWith(groupNames, "/cropped_patches"), 1);
if ~isempty(idx)
    grp = cp.Groups(idx);
    meta.available = true;
    meta.class = "HDF5 group";
    meta.dimensions = "datasets=" + string(numel(grp.Datasets)) + ", groups=" + string(numel(grp.Groups));
end
end

function out = summarizeValue(value)
try
    if iscell(value) && numel(value) == 1
        value = value{1};
    end
    if isnumeric(value) || islogical(value)
        if isempty(value)
            out = "<empty>";
        elseif isscalar(value)
            out = string(value);
        else
            flat = value(:);
            n = min(numel(flat), 10);
            out = "[" + strjoin(string(flat(1:n).'), ", ") + suffixForSize(value, n) + "]";
        end
    elseif ischar(value)
        out = string(value);
    elseif isstring(value)
        out = strjoin(value(:).', ", ");
    elseif iscell(value)
        n = min(numel(value), 5);
        parts = strings(1, n);
        for k = 1:n
            parts(k) = summarizeValue(value{k});
        end
        out = "cell{" + strjoin(parts, ", ") + suffixForSize(value, n) + "}";
    else
        out = "<" + string(class(value)) + " " + string(mat2str(size(value))) + ">";
    end
catch
    out = "<unprintable " + string(class(value)) + ">";
end
end

function suffix = suffixForSize(value, nShown)
if numel(value) > nShown
    suffix = ", ... size=" + string(mat2str(size(value)));
else
    suffix = "";
end
end

function sig = numericSignature(value)
try
    if iscell(value) && numel(value) == 1
        value = value{1};
    end
    nums = double(value(:)).';
    sig = strjoin(compose('%.12g', nums), ',');
catch
    sig = "<non-numeric " + string(class(value)) + " " + string(mat2str(size(value))) + ">";
end
end

function out = scalarToString(value)
if iscell(value) && numel(value) == 1
    value = value{1};
end
if isnumeric(value) || islogical(value)
    if isempty(value)
        out = "";
    else
        out = string(value(1));
    end
elseif ischar(value)
    out = string(value);
elseif isstring(value)
    if isempty(value)
        out = "";
    else
        out = value(1);
    end
else
    out = "<" + string(class(value)) + ">";
end
end

function out = scalarString(value)
if isnumeric(value) && isscalar(value) && isnan(value)
    out = "NaN";
elseif isnumeric(value) || islogical(value)
    out = string(value);
elseif isstring(value)
    out = value;
elseif ischar(value)
    out = string(value);
else
    out = "<" + string(class(value)) + ">";
end
end

function text = joinOrEmpty(values)
if isempty(values)
    text = "<none>";
else
    text = strjoin(string(values), ', ');
end
end

function warn(fid, varargin)
msg = sprintf(varargin{:});
fprintf(fid, 'WARNING: %s\n', msg);
fprintf('WARNING: %s\n', msg);
end
