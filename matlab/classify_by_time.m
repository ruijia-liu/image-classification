function results = classify_by_time()
% Adapted from the original analysis implementation; identifiers and I/O generalized.
%CLASSIFY_BY_TIME Grouped SVM classification separately at each TIME.

cfg = config();
ensureOutputDirs(cfg);
rng(cfg.RANDOM_SEED);

warningFile = fullfile(cfg.paths.logs, 'classification_by_time_warnings.txt');
warningFid = fopen(warningFile, 'w');
if warningFid < 0
    error('Could not open warning log for writing: %s', warningFile);
end
cleanup = onCleanup(@() fclose(warningFid)); %#ok<NASGU>
fprintf(warningFid, 'Classification by TIME warnings\n');
fprintf(warningFid, 'Generated: %s\n\n', datestr(now, 31));

obsTable = load_cached_data("observation");
channels = cfg.expected_channels(:).';
featureNames = "mean_band_" + string(channels);
validateFeatureColumns(obsTable, featureNames);

[predictionTables, foldMetricTables, summaryTables, hyperparamTables, runtimeTables, confusionTables] = ...
    runObservationLevelClassification(obsTable, featureNames, cfg, warningFid);

predictions = vertcatOrEmpty(predictionTables);
fold_metrics = vertcatOrEmpty(foldMetricTables);
time_summary = vertcatOrEmpty(summaryTables);
selected_hyperparameters = vertcatOrEmpty(hyperparamTables);
runtime_info = vertcatOrEmpty(runtimeTables);
confusion_matrices = vertcatOrEmpty(confusionTables);

writeResultTables(cfg, predictions, fold_metrics, time_summary, selected_hyperparameters, runtime_info, confusion_matrices);

patch_results = table();
if runPatchLevelMode(cfg)
    patch_results = runPatchLevelClassification(cfg, featureNames, warningFid);
end

results = struct();
results.predictions = predictions;
results.fold_metrics = fold_metrics;
results.time_summary = time_summary;
results.selected_hyperparameters = selected_hyperparameters;
results.runtime_info = runtime_info;
results.confusion_matrices = confusion_matrices;
results.patch_results = patch_results;
results.warning_file = warningFile;

fprintf('Classification by TIME complete.\n');
fprintf('Prediction rows: %d\n', height(predictions));
fprintf('TIME/model summaries: %d\n', height(time_summary));
fprintf('Results saved to: %s\n', cfg.paths.tables);
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

function validateFeatureColumns(T, featureNames)
missing = setdiff(featureNames, string(T.Properties.VariableNames), 'stable');
if ~isempty(missing)
    error('Input table is missing required feature columns: %s', strjoin(missing, ', '));
end
end

function [predictionTables, foldMetricTables, summaryTables, hyperparamTables, runtimeTables, confusionTables] = ...
    runObservationLevelClassification(obsTable, featureNames, cfg, warningFid)
timeValues = sort(unique(obsTable.time)).';
predictionTables = {};
foldMetricTables = {};
summaryTables = {};
hyperparamTables = {};
runtimeTables = {};
confusionTables = {};

for d = 1:numel(timeValues)
    timeValue = timeValues(d);
    subset = obsTable(obsTable.time == timeValue, :);
    tTime = tic;

    fprintf('Classifying TIME %s (%d observations)\n', string(timeValue), height(subset));
    [canRun, skipReason, timeInfo] = validateTimeSubset(subset);
    warnOnDomainConfounding(subset, timeValue, warningFid);

    if ~canRun
        logWarning(warningFid, 'Skipping TIME %s: %s', string(timeValue), skipReason);
        runtimeTables{end + 1, 1} = makeRuntimeRow(timeValue, "skipped", skipReason, toc(tTime)); %#ok<AGROW>
        continue;
    end

    requestedOuterFolds = getOuterFoldCount(cfg);
    [outerFold, subjectSummary, finalOuterFolds] = make_grouped_folds( ...
        subset.subject_key, subset.analysis_label, requestedOuterFolds, cfg.RANDOM_SEED + round(timeValue));

    models = ["linear_svm", "rbf_svm"];
    for m = 1:numel(models)
        modelType = models(m);
        [predTable, foldMetrics, hyperparams] = runOuterCvForModel( ...
            subset, featureNames, outerFold, finalOuterFolds, modelType, cfg, warningFid);

        pooledMetrics = computeMetrics(predTable.analysis_label, predTable.predicted_label, predTable.positive_score);
        summaryTables{end + 1, 1} = makeSummaryRow(timeValue, modelType, timeInfo, finalOuterFolds, pooledMetrics, toc(tTime)); %#ok<AGROW>
        predictionTables{end + 1, 1} = predTable; %#ok<AGROW>
        foldMetricTables{end + 1, 1} = foldMetrics; %#ok<AGROW>
        hyperparamTables{end + 1, 1} = hyperparams; %#ok<AGROW>
        confusionTables{end + 1, 1} = makeConfusionTable(timeValue, modelType, pooledMetrics.confusion_matrix); %#ok<AGROW>
    end

    runtimeTables{end + 1, 1} = makeRuntimeRow(timeValue, "completed", "", toc(tTime)); %#ok<AGROW>
    fprintf('  TIME %s done in %.2f seconds\n', string(timeValue), toc(tTime));

    if isempty(subjectSummary)
        logWarning(warningFid, 'Internal warning: empty subject summary for TIME %s.', string(timeValue));
    end
end
end

function [canRun, skipReason, info] = validateTimeSubset(T)
labels = ["class1", "class0"];
info = struct();
info.class1_subject_count = countSubjects(T, "class1");
info.class0_subject_count = countSubjects(T, "class0");
info.class1_domains = joinOrNone(unique(T.domain(T.analysis_label == "class1")));
info.class0_domains = joinOrNone(unique(T.domain(T.analysis_label == "class0")));
info.observation_count = height(T);
info.subject_count = numel(unique(T.subject_key));

canRun = true;
skipReason = "";
if height(T) < 6
    canRun = false;
    skipReason = "fewer than 6 observations";
elseif any(~ismember(labels, unique(T.analysis_label)))
    canRun = false;
    skipReason = "only one class is present";
elseif info.class1_subject_count < 2
    canRun = false;
    skipReason = "class1 has fewer than 2 subjects";
elseif info.class0_subject_count < 2
    canRun = false;
    skipReason = "class0 has fewer than 2 subjects";
end
end

function warnOnDomainConfounding(T, timeValue, warningFid)
class1Domains = unique(T.domain(T.analysis_label == "class1"));
class0Domains = unique(T.domain(T.analysis_label == "class0"));
if ~isempty(class1Domains) && ~isempty(class0Domains) && isempty(intersect(class1Domains, class0Domains))
    logWarning(warningFid, 'TIME %s has strong class/domain confounding: class1 domains [%s], class0 domains [%s].', ...
        string(timeValue), joinOrNone(class1Domains), joinOrNone(class0Domains));
end

domains = unique(T.domain);
for e = 1:numel(domains)
    rows = T.domain == domains(e);
    labelsHere = unique(T.analysis_label(rows));
    if numel(labelsHere) == 1
        logWarning(warningFid, 'TIME %s, domain %s contains only class %s.', ...
            string(timeValue), domains(e), labelsHere(1));
    end
end
end

function [predTable, foldMetrics, hyperparams] = runOuterCvForModel(T, featureNames, outerFold, nOuterFolds, modelType, cfg, warningFid)
predictionParts = cell(nOuterFolds, 1);
metricParts = cell(nOuterFolds, 1);
hyperparamParts = cell(nOuterFolds, 1);

for fold = 1:nOuterFolds
    trainRows = outerFold ~= fold;
    testRows = outerFold == fold;

    if numel(unique(T.analysis_label(trainRows))) < 2
        logWarning(warningFid, 'Skipping fold %d for %s at TIME %s because training data has one class.', ...
            fold, modelType, string(T.time(1)));
        continue;
    end
    if numel(unique(T.analysis_label(testRows))) < 2
        logWarning(warningFid, 'Fold %d for %s at TIME %s has one class in the test fold; some metrics may be NaN.', ...
            fold, modelType, string(T.time(1)));
    end

    XTrainRaw = T{trainRows, cellstr(featureNames)};
    XTestRaw = T{testRows, cellstr(featureNames)};
    [XTrain, XTest, mu, sigma] = standardizeTrainTest(XTrainRaw, XTestRaw);
    yTrain = categorical(T.analysis_label(trainRows));

    bestParams = selectHyperparametersInner(T(trainRows, :), featureNames, modelType, cfg, warningFid);
    model = fitSvmModel(XTrain, yTrain, modelType, bestParams);
    [predicted, score] = predict(model, XTest);

    positiveScore = extractPositiveScore(score, model.ClassNames, "class1");
    predictionParts{fold} = makePredictionTable(T(testRows, :), predicted, positiveScore, fold, modelType);

    metrics = computeMetrics(string(T.analysis_label(testRows)), string(predicted), positiveScore);
    metricParts{fold} = makeFoldMetricRow(T.time(find(testRows, 1)), modelType, fold, metrics, bestParams);
    hyperparamParts{fold} = makeHyperparamRow(T.time(find(testRows, 1)), modelType, fold, bestParams, mu, sigma);
end

predTable = vertcatOrEmpty(predictionParts);
foldMetrics = vertcatOrEmpty(metricParts);
hyperparams = vertcatOrEmpty(hyperparamParts);
end

function bestParams = selectHyperparametersInner(trainTable, featureNames, modelType, cfg, warningFid)
grid = makeSvmGrid(modelType, cfg);
requestedInnerFolds = getInnerFoldCount(cfg);

try
    [innerFold, ~, nInnerFolds] = make_grouped_folds(trainTable.subject_key, trainTable.analysis_label, ...
        requestedInnerFolds, cfg.RANDOM_SEED + 99);
catch ME
    logWarning(warningFid, 'Inner grouped folds failed for %s: %s. Using first grid value.', modelType, ME.message);
    bestParams = grid(1);
    return;
end

bestScore = -Inf;
bestParams = grid(1);
for g = 1:numel(grid)
    foldScores = NaN(nInnerFolds, 1);
    for fold = 1:nInnerFolds
        innerTrain = innerFold ~= fold;
        innerVal = innerFold == fold;

        if numel(unique(trainTable.analysis_label(innerTrain))) < 2 || isempty(find(innerVal, 1))
            continue;
        end

        XTrainRaw = trainTable{innerTrain, cellstr(featureNames)};
        XValRaw = trainTable{innerVal, cellstr(featureNames)};
        [XTrain, XVal] = standardizeTrainTest(XTrainRaw, XValRaw);
        yTrain = categorical(trainTable.analysis_label(innerTrain));

        try
            model = fitSvmModel(XTrain, yTrain, modelType, grid(g));
            pred = predict(model, XVal);
            metrics = computeMetrics(string(trainTable.analysis_label(innerVal)), string(pred), NaN(sum(innerVal), 1));
            foldScores(fold) = metrics.balanced_accuracy;
        catch ME
            logWarning(warningFid, 'Inner fold model failed for %s: %s', modelType, ME.message);
        end
    end

    meanScore = mean(foldScores, 'omitnan');
    if isfinite(meanScore) && meanScore > bestScore
        bestScore = meanScore;
        bestParams = grid(g);
    end
end
end

function grid = makeSvmGrid(modelType, cfg)
boxGrid = [0.1, 1, 10];
if isfield(cfg, 'svm') && isfield(cfg.svm, 'BoxConstraint_grid')
    boxGrid = cfg.svm.BoxConstraint_grid;
end

if modelType == "linear_svm"
    grid = repmat(struct('BoxConstraint', 1, 'KernelScale', "none"), numel(boxGrid), 1);
    for i = 1:numel(boxGrid)
        grid(i).BoxConstraint = boxGrid(i);
    end
else
    scaleGrid = {0.5, 1, 2, "auto"};
    if isfield(cfg, 'svm') && isfield(cfg.svm, 'KernelScale_grid_rbf')
        scaleGrid = cfg.svm.KernelScale_grid_rbf;
    end
    grid = repmat(struct('BoxConstraint', 1, 'KernelScale', "auto"), numel(boxGrid) * numel(scaleGrid), 1);
    cursor = 0;
    for b = 1:numel(boxGrid)
        for s = 1:numel(scaleGrid)
            cursor = cursor + 1;
            grid(cursor).BoxConstraint = boxGrid(b);
            grid(cursor).KernelScale = scaleGrid{s};
        end
    end
end
end

function model = fitSvmModel(X, y, modelType, params)
if modelType == "linear_svm"
    model = fitcsvm(X, y, 'KernelFunction', 'linear', ...
        'BoxConstraint', params.BoxConstraint, 'Standardize', false, ...
        'ClassNames', categorical(["class1", "class0"]));
else
    model = fitcsvm(X, y, 'KernelFunction', 'rbf', ...
        'BoxConstraint', params.BoxConstraint, 'KernelScale', params.KernelScale, ...
        'Standardize', false, 'ClassNames', categorical(["class1", "class0"]));
end
end

function [XTrain, XTest, mu, sigma] = standardizeTrainTest(XTrainRaw, XTestRaw)
mu = mean(XTrainRaw, 1, 'omitnan');
sigma = std(XTrainRaw, 0, 1, 'omitnan');
sigma(sigma == 0 | ~isfinite(sigma)) = 1;
XTrain = (XTrainRaw - mu) ./ sigma;
XTest = (XTestRaw - mu) ./ sigma;
end

function positiveScore = extractPositiveScore(score, classNames, positiveClass)
positiveScore = NaN(size(score, 1), 1);
idx = find(string(classNames) == positiveClass, 1);
if ~isempty(idx)
    positiveScore = score(:, idx);
end
end

function predTable = makePredictionTable(T, predicted, positiveScore, fold, modelType)
predTable = table( ...
    T.analysis_label, ...
    string(predicted), ...
    positiveScore, ...
    T.domain, ...
    T.time, ...
    T.subject_key, ...
    T.observation_key, ...
    repmat(fold, height(T), 1), ...
    repmat(modelType, height(T), 1), ...
    'VariableNames', {'analysis_label', 'predicted_label', 'positive_score', ...
    'domain', 'time', 'subject_key', 'observation_key', 'fold', 'model_type'});
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
metrics.specificity = safeDivide(tn, tn + fp);
metrics.precision = safeDivide(tp, tp + fp);
metrics.f1 = safeDivide(2 * metrics.precision * metrics.sensitivity, metrics.precision + metrics.sensitivity);
metrics.balanced_accuracy = mean([metrics.sensitivity, metrics.specificity], 'omitnan');
metrics.tp = tp;
metrics.tn = tn;
metrics.fp = fp;
metrics.fn = fn;
metrics.confusion_matrix = [tp, fn; fp, tn];

if numel(unique(yTrue)) == 2 && all(isfinite(positiveScore))
    try
        [~, ~, ~, auc] = perfcurve(yTrue, positiveScore, positive);
        metrics.roc_auc = auc;
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

function row = makeFoldMetricRow(timeValue, modelType, fold, metrics, params)
row = table(timeValue, modelType, fold, metrics.accuracy, metrics.balanced_accuracy, ...
    metrics.sensitivity, metrics.specificity, metrics.precision, metrics.f1, metrics.roc_auc, ...
    metrics.tp, metrics.tn, metrics.fp, metrics.fn, params.BoxConstraint, string(params.KernelScale), ...
    'VariableNames', {'time', 'model_type', 'fold', 'accuracy', 'balanced_accuracy', ...
    'sensitivity', 'specificity', 'precision', 'f1_score', 'roc_auc', ...
    'tp', 'tn', 'fp', 'fn', 'BoxConstraint', 'KernelScale'});
end

function row = makeSummaryRow(timeValue, modelType, timeInfo, foldCount, metrics, runtimeSeconds)
row = table(timeValue, modelType, timeInfo.class1_subject_count, timeInfo.class0_subject_count, ...
    string(timeInfo.class1_domains), string(timeInfo.class0_domains), foldCount, ...
    metrics.accuracy, metrics.balanced_accuracy, metrics.sensitivity, metrics.specificity, ...
    metrics.precision, metrics.f1, metrics.roc_auc, metrics.tp, metrics.tn, metrics.fp, metrics.fn, runtimeSeconds, ...
    'VariableNames', {'time', 'model_type', 'class1_subject_count', 'class0_subject_count', ...
    'class1_domains', 'class0_domains', 'fold_count', 'accuracy', ...
    'balanced_accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score', ...
    'roc_auc', 'tp', 'tn', 'fp', 'fn', 'runtime_seconds'});
end

function row = makeHyperparamRow(timeValue, modelType, fold, params, mu, sigma)
row = table(timeValue, modelType, fold, params.BoxConstraint, string(params.KernelScale), ...
    string(mat2str(mu, 4)), string(mat2str(sigma, 4)), ...
    'VariableNames', {'time', 'model_type', 'fold', 'BoxConstraint', 'KernelScale', ...
    'training_mean', 'training_sigma'});
end

function row = makeRuntimeRow(timeValue, status, skipReason, runtimeSeconds)
row = table(timeValue, status, string(skipReason), runtimeSeconds, ...
    'VariableNames', {'time', 'status', 'skip_reason', 'runtime_seconds'});
end

function T = makeConfusionTable(timeValue, modelType, cm)
actual = ["class1"; "class1"; "class0"; "class0"];
predicted = ["class1"; "class0"; "class1"; "class0"];
count = [cm(1, 1); cm(1, 2); cm(2, 1); cm(2, 2)];
T = table(repmat(timeValue, 4, 1), repmat(modelType, 4, 1), actual, predicted, count, ...
    'VariableNames', {'time', 'model_type', 'actual_label', 'predicted_label', 'count'});
end

function n = countSubjects(T, label)
rows = T.analysis_label == label;
if any(rows)
    n = numel(unique(T.subject_key(rows)));
else
    n = 0;
end
end

function k = getOuterFoldCount(cfg)
k = 5;
if isfield(cfg, 'speed') && isfield(cfg.speed, 'outer_grouped_folds_fast') && cfg.FAST_MODE
    k = cfg.speed.outer_grouped_folds_fast;
elseif isfield(cfg, 'analysis') && isfield(cfg.analysis, 'outer_grouped_folds')
    k = cfg.analysis.outer_grouped_folds;
end
end

function k = getInnerFoldCount(cfg)
k = 3;
if isfield(cfg, 'speed') && isfield(cfg.speed, 'inner_grouped_folds_fast') && cfg.FAST_MODE
    k = cfg.speed.inner_grouped_folds_fast;
elseif isfield(cfg, 'analysis') && isfield(cfg.analysis, 'inner_grouped_folds')
    k = cfg.analysis.inner_grouped_folds;
end
end

function tf = runPatchLevelMode(cfg)
tf = false;
if isfield(cfg, 'RUN_PATCH_LEVEL_CLASSIFICATION')
    tf = logical(cfg.RUN_PATCH_LEVEL_CLASSIFICATION);
elseif isfield(cfg, 'analysis') && isfield(cfg.analysis, 'run_patch_level_classification')
    tf = logical(cfg.analysis.run_patch_level_classification);
end
end

function patchResults = runPatchLevelClassification(cfg, featureNames, warningFid)
logWarning(warningFid, 'Optional patch-level mode is enabled. Loading patch_table.mat with capped reproducible sampling.');
patchTable = load_cached_data("patch");
maxPatches = cfg.analysis.max_patches_per_observation;
if isfield(cfg, 'MAX_PATCHES_PER_OBSERVATION')
    maxPatches = cfg.MAX_PATCHES_PER_OBSERVATION;
end
sampled = samplePatchTableByObservation(patchTable, maxPatches, cfg.RANDOM_SEED);
patchResults = table(height(patchTable), height(sampled), maxPatches, ...
    'VariableNames', {'original_patch_rows', 'sampled_patch_rows', 'max_patches_per_observation'});
if isempty(intersect(featureNames, string(sampled.Properties.VariableNames)))
    logWarning(warningFid, 'Patch-level table does not use observation feature names; patch-level classification is not run by this script.');
end
end

function sampled = samplePatchTableByObservation(patchTable, maxPatches, seed)
rng(seed);
obsKeys = unique(patchTable.observation_key, 'stable');
keep = false(height(patchTable), 1);
for i = 1:numel(obsKeys)
    rows = find(patchTable.observation_key == obsKeys(i));
    if numel(rows) <= maxPatches
        keep(rows) = true;
    else
        rows = rows(randperm(numel(rows), maxPatches));
        keep(rows) = true;
    end
end
sampled = patchTable(keep, :);
end

function writeResultTables(cfg, predictions, foldMetrics, timeSummary, hyperparams, runtimeInfo, confusionMatrices)
writetable(predictions, fullfile(cfg.paths.predictions, 'classification_by_time_oof_predictions.csv'));
writetable(foldMetrics, fullfile(cfg.paths.tables, 'classification_by_time_fold_metrics.csv'));
writetable(timeSummary, fullfile(cfg.paths.tables, 'classification_by_time_summary.csv'));
writetable(hyperparams, fullfile(cfg.paths.tables, 'classification_by_time_selected_hyperparameters.csv'));
writetable(runtimeInfo, fullfile(cfg.paths.tables, 'classification_by_time_runtime.csv'));
writetable(confusionMatrices, fullfile(cfg.paths.tables, 'classification_by_time_confusion_matrices.csv'));

save(fullfile(cfg.paths.cache, 'classification_by_time_results.mat'), ...
    'predictions', 'foldMetrics', 'timeSummary', 'hyperparams', 'runtimeInfo', 'confusionMatrices', '-v7.3');
end

function out = vertcatOrEmpty(parts)
parts = parts(~cellfun(@isempty, parts));
if isempty(parts)
    out = table();
else
    out = vertcat(parts{:});
end
end

function text = joinOrNone(values)
if isempty(values)
    text = "<none>";
else
    text = strjoin(string(values(:).'), ', ');
end
end

function logWarning(fid, varargin)
fprintf(fid, 'WARNING: %s\n', sprintf(varargin{:}));
end
