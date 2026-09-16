function results = evaluate_source_level(predictionFile)
% Adapted from the original analysis implementation; identifiers and I/O generalized.
%EVALUATE_SOURCE_LEVEL Aggregate patch-level OOF predictions to observations.
%
% Expected patch-level prediction columns:
% analysis_label, predicted_label, positive_score, domain, time,
% subject_key, observation_key, fold, model_type.

cfg = config();
ensureOutputDirs(cfg);

if nargin < 1 || strlength(string(predictionFile)) == 0
    predictionFile = fullfile(cfg.paths.predictions, 'classification_by_time_patch_oof_predictions.csv');
end
predictionFile = string(predictionFile);

warningFile = fullfile(cfg.paths.logs, 'source_level_evaluation_warnings.txt');
warningFid = fopen(warningFile, 'w');
if warningFid < 0
    error('Could not open warning log for writing: %s', warningFile);
end
cleanup = onCleanup(@() fclose(warningFid)); %#ok<NASGU>
fprintf(warningFid, 'Source/observation-level aggregation warnings\n');
fprintf(warningFid, 'Generated: %s\n\n', datestr(now, 31));

if exist(predictionFile, 'file') ~= 2
    error(['Patch-level prediction file not found: %s\n', ...
        'This evaluator requires patch-level out-of-fold predictions. ', ...
        'Run a patch-level classifier that writes classification_by_time_patch_oof_predictions.csv, ', ...
        'or pass an explicit patch-level prediction CSV to evaluate_source_level(predictionFile).'], predictionFile);
end

predictions = readtable(predictionFile, 'TextType', 'string');
validatePredictionTable(predictions);
predictions = normalizePredictionTable(predictions);

if ~ismember("patch_key", string(predictions.Properties.VariableNames))
    error(['Prediction file does not contain patch_key: %s\n', ...
        'Refusing to aggregate observation-level predictions as patch-level votes.'], predictionFile);
end

threshold = getScoreThreshold(cfg);
fewPatchThreshold = getFewPatchThreshold(cfg);

[observationPredictions, observationMetrics, comparisonTable] = aggregateAndEvaluate( ...
    predictions, threshold, fewPatchThreshold, warningFid);

writeOutputs(cfg, observationPredictions, observationMetrics, comparisonTable);

results = struct();
results.prediction_file = predictionFile;
results.observation_predictions = observationPredictions;
results.observation_metrics = observationMetrics;
results.patch_vs_observation_comparison = comparisonTable;
results.warning_file = warningFile;

fprintf('Source/observation-level evaluation complete.\n');
fprintf('Input predictions: %s\n', predictionFile);
fprintf('Observation rows: %d\n', height(observationPredictions));
fprintf('Metrics rows: %d\n', height(observationMetrics));
fprintf('Outputs saved to: %s\n', cfg.paths.tables);
fprintf('Warnings saved to: %s\n', warningFile);

end

function ensureOutputDirs(cfg)
dirs = {cfg.paths.tables, cfg.paths.logs, cfg.paths.predictions};
for k = 1:numel(dirs)
    if ~exist(dirs{k}, 'dir')
        mkdir(dirs{k});
    end
end
end

function validatePredictionTable(T)
required = ["analysis_label", "predicted_label", "positive_score", "domain", ...
    "time", "subject_key", "observation_key", "fold", "model_type"];
missing = setdiff(required, string(T.Properties.VariableNames), 'stable');
if ~isempty(missing)
    error('Prediction table is missing required columns: %s', strjoin(missing, ', '));
end
end

function T = normalizePredictionTable(T)
stringVars = ["analysis_label", "predicted_label", "domain", "subject_key", ...
    "observation_key", "model_type"];
for k = 1:numel(stringVars)
    if ismember(stringVars(k), string(T.Properties.VariableNames))
        T.(stringVars(k)) = string(T.(stringVars(k)));
    end
end
T.positive_score = double(T.positive_score);
T.time = double(T.time);
T.fold = double(T.fold);
end

function threshold = getScoreThreshold(cfg)
threshold = 0;
if isfield(cfg, 'analysis') && isfield(cfg.analysis, 'score_threshold')
    threshold = cfg.analysis.score_threshold;
end
end

function threshold = getFewPatchThreshold(cfg)
threshold = 10;
if isfield(cfg, 'analysis') && isfield(cfg.analysis, 'few_patch_threshold')
    threshold = cfg.analysis.few_patch_threshold;
end
end

function [observationPredictions, observationMetrics, comparisonTable] = aggregateAndEvaluate(T, threshold, fewPatchThreshold, warningFid)
groupVars = ["model_type", "time", "observation_key"];
[g, keyTable] = findgroups(T(:, cellstr(groupVars)));
nGroups = height(keyTable);

rows = cell(nGroups, 1);
for i = 1:nGroups
    groupRows = g == i;
    part = T(groupRows, :);
    rows{i} = aggregateOneObservation(part, threshold, fewPatchThreshold, warningFid);
end
observationPredictions = vertcat(rows{:});

observationMetrics = computeGroupedObservationMetrics(observationPredictions);
patchMetrics = computeGroupedPatchMetrics(T);
comparisonTable = comparePatchAndObservationMetrics(patchMetrics, observationMetrics);
end

function row = aggregateOneObservation(T, threshold, fewPatchThreshold, warningFid)
assert(all(isfinite(T.positive_score)), 'Scores must be finite.');
assert(numel(unique(T.patch_key)) == height(T), 'Duplicate patch predictions.');
assert(numel(unique(T.analysis_label)) == 1, 'Conflicting observation labels.');
assert(numel(unique(T.fold)) == 1, 'Observation predictions span multiple outer folds.');
assert(numel(unique(T.subject_key)) == 1, 'Observation spans multiple sources.');
labels = unique(T.analysis_label);
if numel(labels) ~= 1
    logWarning(warningFid, 'Observation %s has inconsistent true labels: %s', ...
        T.observation_key(1), strjoin(labels, ', '));
end

folds = unique(T.fold);
if numel(folds) ~= 1
    logWarning(warningFid, ['Observation %s has predictions from multiple outer folds: %s. ', ...
        'This suggests fold mixing and should be checked.'], T.observation_key(1), mat2str(folds.'));
end

domains = unique(T.domain);
subjects = unique(T.subject_key);
if numel(domains) ~= 1
    logWarning(warningFid, 'Observation %s has inconsistent domains: %s', ...
        T.observation_key(1), strjoin(domains, ', '));
end
if numel(subjects) ~= 1
    logWarning(warningFid, 'Observation %s has inconsistent subject_key values: %s', ...
        T.observation_key(1), strjoin(subjects, ', '));
end

nPatches = height(T);
meanScore = mean(T.positive_score, 'omitnan');
majorityPred = majorityVote(T.predicted_label);
meanScorePred = labelFromScore(meanScore, threshold);
fewPatchFlag = nPatches < fewPatchThreshold;

row = table( ...
    T.model_type(1), ...
    T.time(1), ...
    domains(1), ...
    subjects(1), ...
    T.observation_key(1), ...
    folds(1), ...
    labels(1), ...
    nPatches, ...
    meanScore, ...
    meanScorePred, ...
    majorityPred, ...
    fewPatchFlag, ...
    threshold, ...
    'VariableNames', {'model_type', 'time', 'domain', 'subject_key', ...
    'observation_key', 'fold', 'true_label', 'num_patches_contributing', ...
    'mean_positive_score', 'mean_score_prediction', ...
    'majority_vote_prediction', 'few_patches_flag', 'threshold'});
end

function label = majorityVote(predictedLabels)
labels = unique(predictedLabels);
counts = zeros(numel(labels), 1);
for i = 1:numel(labels)
    counts(i) = sum(predictedLabels == labels(i));
end
[maxCount, idx] = max(counts);
if sum(counts == maxCount) > 1
    if any(labels == "class1")
        label = "class1";
    else
        label = labels(idx);
    end
else
    label = labels(idx);
end
end

function label = labelFromScore(score, threshold)
if score >= threshold
    label = "class1";
else
    label = "class0";
end
end

function metricsTable = computeGroupedObservationMetrics(T)
methods = ["mean_score", "majority_vote"];
rows = {};
cursor = 0;
models = unique(T.model_type).';
timeValues = sort(unique(T.time)).';

for m = 1:numel(models)
    for d = 1:numel(timeValues)
        baseRows = T.model_type == models(m) & T.time == timeValues(d);
        if ~any(baseRows)
            continue;
        end
        for a = 1:numel(methods)
            method = methods(a);
            if method == "mean_score"
                pred = T.mean_score_prediction(baseRows);
                score = T.mean_positive_score(baseRows);
            else
                pred = T.majority_vote_prediction(baseRows);
                score = T.mean_positive_score(baseRows);
            end
            trueLabel = T.true_label(baseRows);
            metrics = computeMetrics(trueLabel, pred, score);
            cursor = cursor + 1;
            rows{cursor, 1} = makeMetricRow(models(m), timeValues(d), method, metrics, sum(baseRows)); %#ok<AGROW>
        end
    end
end

metricsTable = vertcat(rows{:});
end

function metricsTable = computeGroupedPatchMetrics(T)
rows = {};
cursor = 0;
models = unique(T.model_type).';
timeValues = sort(unique(T.time)).';

for m = 1:numel(models)
    for d = 1:numel(timeValues)
        rowsHere = T.model_type == models(m) & T.time == timeValues(d);
        if ~any(rowsHere)
            continue;
        end
        metrics = computeMetrics(T.analysis_label(rowsHere), T.predicted_label(rowsHere), T.positive_score(rowsHere));
        cursor = cursor + 1;
        rows{cursor, 1} = makeMetricRow(models(m), timeValues(d), "patch_level", metrics, sum(rowsHere)); %#ok<AGROW>
    end
end

if isempty(rows)
    metricsTable = table();
else
    metricsTable = vertcat(rows{:});
end
end

function comparisonTable = comparePatchAndObservationMetrics(patchMetrics, observationMetrics)
if isempty(patchMetrics) || isempty(observationMetrics)
    comparisonTable = table();
    return;
end

patchMetrics.Properties.VariableNames = "patch_" + string(patchMetrics.Properties.VariableNames);
observationMetrics.Properties.VariableNames = "observation_" + string(observationMetrics.Properties.VariableNames);

patchMetrics.patch_time = double(patchMetrics.patch_time);
observationMetrics.observation_time = double(observationMetrics.observation_time);

comparisonTable = innerjoin(patchMetrics, observationMetrics, ...
    'LeftKeys', {'patch_model_type', 'patch_time'}, ...
    'RightKeys', {'observation_model_type', 'observation_time'});
end

function row = makeMetricRow(modelType, timeValue, aggregationMethod, metrics, nRows)
row = table(modelType, timeValue, aggregationMethod, nRows, metrics.accuracy, ...
    metrics.balanced_accuracy, metrics.sensitivity, metrics.specificity, ...
    metrics.precision, metrics.recall, metrics.f1, metrics.roc_auc, ...
    metrics.tp, metrics.tn, metrics.fp, metrics.fn, ...
    'VariableNames', {'model_type', 'time', 'aggregation_method', 'n_rows', ...
    'accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', ...
    'precision', 'recall', 'f1_score', 'roc_auc', 'tp', 'tn', 'fp', 'fn'});
end

function metrics = computeMetrics(yTrue, yPred, positiveScore)
yTrue = string(yTrue(:));
yPred = string(yPred(:));
positive = "class1";
negative = "class0";

tp = sum(yTrue == positive & yPred == positive);
tn = sum(yTrue == negative & yPred == negative);
fp = sum(yTrue == negative & yPred == positive);
fn = sum(yTrue == positive & yPred == negative);

metrics = struct();
metrics.accuracy = safeDivide(tp + tn, numel(yTrue));
metrics.sensitivity = safeDivide(tp, tp + fn);
metrics.recall = metrics.sensitivity;
metrics.specificity = safeDivide(tn, tn + fp);
metrics.precision = safeDivide(tp, tp + fp);
metrics.f1 = safeDivide(2 * tp, 2 * tp + fp + fn);
metrics.balanced_accuracy = mean([metrics.sensitivity, metrics.specificity], 'omitnan');
metrics.tp = tp;
metrics.tn = tn;
metrics.fp = fp;
metrics.fn = fn;

if numel(unique(yTrue)) == 2 && all(isfinite(positiveScore))
    try
        [~, ~, ~, metrics.roc_auc] = perfcurve(yTrue, positiveScore, positive);
    catch
        metrics.roc_auc = NaN;
    end
else
    metrics.roc_auc = NaN;
end
end

function value = safeDivide(a, b)
if b == 0
    value = NaN;
else
    value = a / b;
end
end

function writeOutputs(cfg, observationPredictions, observationMetrics, comparisonTable)
writetable(observationPredictions, fullfile(cfg.paths.predictions, 'source_level_observation_predictions.csv'));
writetable(observationMetrics, fullfile(cfg.paths.tables, 'source_level_observation_metrics.csv'));
writetable(comparisonTable, fullfile(cfg.paths.tables, 'source_level_patch_vs_observation_metrics.csv'));

save(fullfile(cfg.paths.cache, 'source_level_evaluation.mat'), ...
    'observationPredictions', 'observationMetrics', 'comparisonTable', '-v7.3');
end

function logWarning(fid, varargin)
fprintf(fid, 'WARNING: %s\n', sprintf(varargin{:}));
end
