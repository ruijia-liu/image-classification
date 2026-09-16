function quality = check_data_quality()
% Adapted from the original analysis implementation; identifiers and I/O generalized.
%CHECK_DATA_QUALITY Summarise cached image feature tables and QA checks.

cfg = config();
ensureOutputDirs(cfg);

warningFile = fullfile(cfg.paths.logs, 'data_quality_warnings.txt');
fid = fopen(warningFile, 'w');
if fid < 0
    error('Could not open warning log for writing: %s', warningFile);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

fprintf(fid, 'Image cached data quality warnings\n');
fprintf(fid, 'Generated: %s\n\n', datestr(now, 31));

[patchTable, obsTable] = load_cached_data("both");
bandNames = "band_" + string(cfg.expected_channels);
meanBandNames = "mean_" + bandNames;

validateInputs(patchTable, obsTable, bandNames, meanBandNames);

quality = struct();
quality.warning_file = warningFile;
quality.patch_rows = height(patchTable);
quality.observation_rows = height(obsTable);
quality.subject_count = numel(unique(obsTable.subject_key));

fprintf('\nRunning data quality checks on cached tables only...\n');
fprintf('Patches: %d | observations: %d | physical subjects: %d\n', ...
    quality.patch_rows, quality.observation_rows, quality.subject_count);

summaries = makeSummaries(patchTable, obsTable, bandNames);
quality.summaries = summaries;
writeSummaries(cfg, summaries);

runQualityChecks(cfg, patchTable, obsTable, bandNames, meanBandNames, fid);
makeFigures(cfg, patchTable, obsTable, bandNames);

fprintf('Data quality summaries saved to: %s\n', cfg.paths.tables);
fprintf('Data quality warnings saved to: %s\n', warningFile);
fprintf('Data quality figures saved to: %s\n', cfg.paths.figures);

end

function ensureOutputDirs(cfg)
dirs = {cfg.paths.tables, cfg.paths.logs, cfg.paths.figures};
for k = 1:numel(dirs)
    if ~exist(dirs{k}, 'dir')
        mkdir(dirs{k});
    end
end
end

function validateInputs(patchTable, obsTable, bandNames, meanBandNames)
requiredPatch = ["source_file", "dataset_group", "original_class", "analysis_label", ...
    "domain", "time", "object_number", "object_name", "subject_key", "observation_key", ...
    "patch_index", "patch_key", "location_y", "location_x", "patch_size", bandNames];
requiredObs = ["source_file", "dataset_group", "original_class", "analysis_label", ...
    "domain", "time", "object_number", "object_name", "subject_key", "observation_key", ...
    "num_patches", "patch_size", meanBandNames];
assertColumns(patchTable, requiredPatch, 'patch table');
assertColumns(obsTable, requiredObs, 'observation table');
end

function assertColumns(tbl, required, tableName)
missing = setdiff(required, string(tbl.Properties.VariableNames), 'stable');
if ~isempty(missing)
    error('Cached %s is missing required columns: %s', tableName, strjoin(missing, ', '));
end
end

function summaries = makeSummaries(patchTable, obsTable, bandNames)
summaries = struct();

summaries.by_analysis_label = groupSummary(obsTable, patchTable, "analysis_label");
summaries.by_domain = groupSummary(obsTable, patchTable, "domain");
summaries.by_time = groupSummary(obsTable, patchTable, "time");
summaries.by_domain_time_label = groupSummary(obsTable, patchTable, ["domain", "time", "analysis_label"]);
summaries.subject_counts_by_domain_time_label = subjectCountSummary(obsTable, ["domain", "time", "analysis_label"]);
summaries.observation_counts = groupcounts(obsTable, ["analysis_label", "domain", "time"]);
summaries.patch_counts = groupcounts(patchTable, ["analysis_label", "domain", "time"]);
summaries.patches_per_observation = patchesPerObservationSummary(obsTable);
summaries.repeated_measurements_per_subject = repeatedMeasurementsSummary(obsTable);
summaries.missing_values_patch_table = missingByColumn(patchTable);
summaries.missing_values_observation_table = missingByColumn(obsTable);
summaries.band_summary_patch_table = bandSummary(patchTable, bandNames);
end

function out = groupSummary(obsTable, patchTable, groupVars)
[gObs, keysObs] = findgroups(obsTable(:, cellstr(groupVars)));
observation_count = splitapply(@numel, obsTable.observation_key, gObs);
subject_count = splitapply(@(x) numel(unique(x)), obsTable.subject_key, gObs);
patch_count = splitapply(@sum, obsTable.num_patches, gObs);
mean_patches_per_observation = splitapply(@mean, obsTable.num_patches, gObs);
median_patches_per_observation = splitapply(@median, obsTable.num_patches, gObs);
out = keysObs;
out.observation_count = observation_count;
out.subject_count = subject_count;
out.patch_count = patch_count;
out.mean_patches_per_observation = mean_patches_per_observation;
out.median_patches_per_observation = median_patches_per_observation;

if nargin > 1 && ~isempty(patchTable)
    patchRows = groupcounts(patchTable, groupVars);
    patchRows.Properties.VariableNames(end) = "patch_table_row_count";
    out = outerjoin(out, patchRows, 'Keys', cellstr(groupVars), 'MergeKeys', true);
end
end

function out = subjectCountSummary(obsTable, groupVars)
[g, keys] = findgroups(obsTable(:, cellstr(groupVars)));
unique_subject_count = splitapply(@(x) numel(unique(x)), obsTable.subject_key, g);
out = keys;
out.unique_subject_count = unique_subject_count;
end

function out = patchesPerObservationSummary(obsTable)
out = table();
out.observation_count = height(obsTable);
out.min_patches = min(obsTable.num_patches);
out.q1_patches = quantile(obsTable.num_patches, 0.25);
out.median_patches = median(obsTable.num_patches);
out.mean_patches = mean(obsTable.num_patches);
out.q3_patches = quantile(obsTable.num_patches, 0.75);
out.max_patches = max(obsTable.num_patches);
end

function out = repeatedMeasurementsSummary(obsTable)
[g, keys] = findgroups(obsTable(:, {'subject_key', 'analysis_label', 'domain', 'object_number'}));
observation_count = splitapply(@numel, obsTable.observation_key, g);
unique_time_count = splitapply(@(x) numel(unique(x)), obsTable.time, g);
total_patch_count = splitapply(@sum, obsTable.num_patches, g);
out = keys;
out.observation_count = observation_count;
out.unique_time_count = unique_time_count;
out.total_patch_count = total_patch_count;
out = sortrows(out, {'observation_count', 'unique_time_count'}, {'descend', 'descend'});
end

function out = missingByColumn(tbl)
names = string(tbl.Properties.VariableNames).';
missing_count = zeros(numel(names), 1);
missing_fraction = zeros(numel(names), 1);
for k = 1:numel(names)
    values = tbl.(names(k));
    isMissing = ismissing(values);
    missing_count(k) = sum(isMissing);
    missing_fraction(k) = missing_count(k) / max(height(tbl), 1);
end
out = table(names, missing_count, missing_fraction, ...
    'VariableNames', {'column_name', 'missing_count', 'missing_fraction'});
end

function out = bandSummary(patchTable, bandNames)
out = table();
out.band = bandNames(:);
out.mean_value = zeros(numel(bandNames), 1);
out.std_value = zeros(numel(bandNames), 1);
out.min_value = zeros(numel(bandNames), 1);
out.max_value = zeros(numel(bandNames), 1);
out.nonfinite_count = zeros(numel(bandNames), 1);
for k = 1:numel(bandNames)
    x = patchTable.(bandNames(k));
    finiteX = x(isfinite(x));
    out.mean_value(k) = mean(finiteX, 'omitnan');
    out.std_value(k) = std(finiteX, 0, 'omitnan');
    out.min_value(k) = min(finiteX);
    out.max_value(k) = max(finiteX);
    out.nonfinite_count(k) = sum(~isfinite(x));
end
end

function writeSummaries(cfg, summaries)
names = fieldnames(summaries);
for k = 1:numel(names)
    outFile = fullfile(cfg.paths.tables, "data_quality_" + string(names{k}) + ".csv");
    writetable(summaries.(names{k}), outFile);
end
end

function runQualityChecks(cfg, patchTable, obsTable, bandNames, meanBandNames, fid)
checkDuplicateKeys(patchTable, obsTable, fid);
checkSubjectConsistency(obsTable, fid);
checkOriginalClassConsistency(obsTable, fid);
checkSubjectsInBothGroups(obsTable, fid);
checkPatchCounts(obsTable, fid);
checkBandValues(patchTable, obsTable, bandNames, meanBandNames, fid);
checkChannels(cfg, bandNames, fid);
checkTimeClassDomainCoverage(obsTable, fid);
checkConfounding(obsTable, fid);
end

function checkDuplicateKeys(patchTable, obsTable, fid)
if numel(unique(patchTable.patch_key)) ~= height(patchTable)
    logWarning(fid, 'Duplicated patch_key values detected.');
end
if numel(unique(obsTable.observation_key)) ~= height(obsTable)
    logWarning(fid, 'Duplicated observation_key values detected.');
end
end

function checkSubjectConsistency(obsTable, fid)
[g, keys] = findgroups(obsTable.subject_key);
labelCounts = splitapply(@(x) numel(unique(x)), obsTable.analysis_label, g);
bad = labelCounts > 1;
if any(bad)
    logWarning(fid, 'Inconsistent analysis_label within subject_key: %s', strjoin(keys(bad), ', '));
end
end

function checkOriginalClassConsistency(obsTable, fid)
[g, keys] = findgroups(obsTable.subject_key);
classCounts = splitapply(@(x) numel(unique(x)), obsTable.original_class, g);
bad = classCounts > 1;
if any(bad)
    logWarning(fid, 'Inconsistent original_class values within subject_key: %s', strjoin(keys(bad), ', '));
end
end

function checkSubjectsInBothGroups(obsTable, fid)
physicalKey = obsTable.domain + "_" + string(obsTable.object_number);
[g, keys] = findgroups(physicalKey);
groupCounts = splitapply(@(x) numel(unique(x)), obsTable.dataset_group, g);
bad = groupCounts > 1;
if any(bad)
    examples = keys(bad);
    logWarning(fid, 'Physical domain_object_number keys appear in both class1 and class0 groups. First examples: %s', ...
        strjoin(examples(1:min(numel(examples), 20)), ', '));
end
end

function checkPatchCounts(obsTable, fid)
fewThreshold = 10;
fewRows = obsTable.num_patches < fewThreshold;
if any(fewRows)
    logWarning(fid, '%d observations have fewer than %d patches.', sum(fewRows), fewThreshold);
end

q1 = quantile(obsTable.num_patches, 0.25);
q3 = quantile(obsTable.num_patches, 0.75);
iqrValue = q3 - q1;
low = q1 - 1.5 * iqrValue;
high = q3 + 1.5 * iqrValue;
unusual = obsTable.num_patches < low | obsTable.num_patches > high;
if any(unusual)
    logWarning(fid, '%d observations have unusually large or small patch counts by 1.5 IQR rule. Range flagged: < %.2f or > %.2f.', ...
        sum(unusual), low, high);
end
end

function checkBandValues(patchTable, obsTable, bandNames, meanBandNames, fid)
for k = 1:numel(bandNames)
    x = patchTable.(bandNames(k));
    if any(~isfinite(x))
        logWarning(fid, 'Non-finite patch band values detected in %s: %d rows.', bandNames(k), sum(~isfinite(x)));
    end
    sx = std(x, 0, 'omitnan');
    if sx == 0
        logWarning(fid, 'Constant patch band detected: %s.', bandNames(k));
    elseif sx < 1e-10
        logWarning(fid, 'Nearly constant patch band detected: %s, std=%g.', bandNames(k), sx);
    end
end

for k = 1:numel(meanBandNames)
    x = obsTable.(meanBandNames(k));
    if any(~isfinite(x))
        logWarning(fid, 'Non-finite observation band values detected in %s: %d rows.', meanBandNames(k), sum(~isfinite(x)));
    end
end
end

function checkChannels(cfg, bandNames, fid)
expectedBandNames = "band_" + string(cfg.expected_channels);
if ~isequal(bandNames(:), expectedBandNames(:))
    logWarning(fid, 'Band columns do not match cfg.expected_channels.');
end

end

function checkTimeClassDomainCoverage(obsTable, fid)
timeByClass = groupcounts(obsTable, ["time", "analysis_label"]);
allLabels = unique(obsTable.analysis_label);
for timeValue = unique(obsTable.time).'
    rows = timeByClass.time == timeValue;
    labelsHere = unique(timeByClass.analysis_label(rows));
    missing = setdiff(allLabels, labelsHere);
    if ~isempty(missing)
        logWarning(fid, 'TIME %s occurs only in one class or is missing class(es): %s.', ...
            string(timeValue), strjoin(missing, ', '));
    end
end

domainByTime = groupcounts(obsTable, ["time", "domain"]);
for timeValue = unique(obsTable.time).'
    rows = domainByTime.time == timeValue;
    domainsHere = unique(domainByTime.domain(rows));
    if numel(domainsHere) == 1
        logWarning(fid, 'TIME %s occurs only in domain %s.', string(timeValue), domainsHere(1));
    end
end
end

function checkConfounding(obsTable, fid)
classByDomain = groupcounts(obsTable, ["domain", "analysis_label"]);
for expName = unique(obsTable.domain).'
    rows = classByDomain.domain == expName;
    labelsHere = unique(classByDomain.analysis_label(rows));
    if numel(labelsHere) == 1
        logWarning(fid, 'Domain %s contains only analysis_label %s.', expName, labelsHere(1));
    end
end

timeByDomain = groupcounts(obsTable, ["domain", "time"]);
for expName = unique(obsTable.domain).'
    rows = timeByDomain.domain == expName;
    timeHere = unique(timeByDomain.time(rows));
    if numel(timeHere) == 1
        logWarning(fid, 'Domain %s contains only TIME %s.', expName, string(timeHere(1)));
    end
end

domainByTime = groupcounts(obsTable, ["time", "domain"]);
for timeValue = unique(obsTable.time).'
    rows = domainByTime.time == timeValue;
    expHere = unique(domainByTime.domain(rows));
    if numel(expHere) == 1
        logWarning(fid, 'TIME %s is confounded with domain %s.', string(timeValue), expHere(1));
    end
end
end

function makeFigures(cfg, patchTable, obsTable, bandNames)
fig1 = figure('Visible', 'off');
histogram(obsTable.num_patches);
xlabel('Patches per observation');
ylabel('Number of observations');
title('Patches per observation');
grid on;
saveas(fig1, fullfile(cfg.paths.figures, 'data_quality_patches_per_observation.png'));
close(fig1);

subjectCounts = groupcounts(obsTable, "subject_key");
fig2 = figure('Visible', 'off');
histogram(subjectCounts.GroupCount);
xlabel('Observations per subject');
ylabel('Number of subjects');
title('Repeated observations per subject');
grid on;
saveas(fig2, fullfile(cfg.paths.figures, 'data_quality_observations_per_subject.png'));
close(fig2);

fig3 = figure('Visible', 'off');
bandMatrix = patchTable{:, cellstr(bandNames)};
boxplot(bandMatrix, 'Labels', cellstr(erase(bandNames, "band_")));
xlabel('Channel');
ylabel('Patch mean value');
title('Patch-level band value distributions');
grid on;
saveas(fig3, fullfile(cfg.paths.figures, 'data_quality_band_boxplots.png'));
close(fig3);
end

function logWarning(fid, varargin)
fprintf(fid, 'WARNING: %s\n', sprintf(varargin{:}));
end
