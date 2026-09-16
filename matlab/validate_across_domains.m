function results = validate_across_domains()
% Adapted from the original analysis implementation; identifiers and I/O generalized.
%VALIDATE_ACROSS_DOMAINS Leave-one-domain-out SVM validation.

cfg = config();
ensureOutputDirs(cfg);
rng(cfg.RANDOM_SEED);

warningFile = fullfile(cfg.paths.logs, 'validate_across_domains_warnings.txt');
warningFid = fopen(warningFile, 'w');
if warningFid < 0
    error('Could not open warning log for writing: %s', warningFile);
end
cleanup = onCleanup(@() fclose(warningFid)); %#ok<NASGU>
fprintf(warningFid, 'Leave-one-domain-out validation warnings\n');
fprintf(warningFid, 'Generated: %s\n\n', datestr(now, 31));

obsTable = load_cached_data("observation");
channels = cfg.expected_channels(:).';
featureNames = "mean_band_" + string(channels);
validateFeatureColumns(obsTable, featureNames);

domains = sort(unique(obsTable.domain)).';
models = ["linear_svm", "rbf_svm"];

predictionParts = {};
summaryParts = {};
hyperparamParts = {};
runtimeParts = {};

for e = 1:numel(domains)
    heldOutDomain = domains(e);
    tDomain = tic;
    fprintf('Held-out domain: %s\n', heldOutDomain);

    testRows = obsTable.domain == heldOutDomain;
    trainRows = ~testRows;
    trainTable = obsTable(trainRows, :);
    testTable = obsTable(testRows, :);
    trainingDomains = joinOrNone(sort(unique(trainTable.domain)));

    checkSubjectLeakage(trainTable, testTable, heldOutDomain);
    domainInfo = describeDomainSplit(trainTable, testTable);

    if numel(unique(trainTable.analysis_label)) < 2
        logWarning(warningFid, 'Skipping held-out domain %s: training data has fewer than two classes.', heldOutDomain);
        runtimeParts{end + 1, 1} = makeRuntimeRow(heldOutDomain, "skipped", "training data has fewer than two classes", toc(tDomain)); %#ok<AGROW>
        continue;
    end

    for m = 1:numel(models)
        modelType = models(m);
        [predTable, summaryRow, hyperRow] = runOneHeldOutDomain( ...
            trainTable, testTable, featureNames, heldOutDomain, trainingDomains, ...
            modelType, domainInfo, cfg, warningFid);

        predictionParts{end + 1, 1} = predTable; %#ok<AGROW>
        summaryParts{end + 1, 1} = summaryRow; %#ok<AGROW>
        hyperparamParts{end + 1, 1} = hyperRow; %#ok<AGROW>
    end

    runtimeParts{end + 1, 1} = makeRuntimeRow(heldOutDomain, "completed", "", toc(tDomain)); %#ok<AGROW>
    fprintf('  completed in %.2f seconds\n', toc(tDomain));
end

predictions = vertcatOrEmpty(predictionParts);
summaryTable = vertcatOrEmpty(summaryParts);
hyperparameterTable = vertcatOrEmpty(hyperparamParts);
runtimeTable = vertcatOrEmpty(runtimeParts);

writeOutputs(cfg, predictions, summaryTable, hyperparameterTable, runtimeTable);

if runPatchLevelMode(cfg)
    logWarning(warningFid, 'Patch-level leave-one-domain-out mode is enabled in config, but this script currently uses observation-level features only.');
end

results = struct();
results.predictions = predictions;
results.summary = summaryTable;
results.selected_hyperparameters = hyperparameterTable;
results.runtime = runtimeTable;
results.warning_file = warningFile;

fprintf('Leave-one-domain-out validation complete.\n');
fprintf('Prediction rows: %d\n', height(predictions));
fprintf('Summary rows: %d\n', height(summaryTable));
fprintf('Results saved to: %s\n', cfg.paths.tables);
fprintf('Predictions saved to: %s\n', cfg.paths.predictions);
fprintf('Warnings saved to: %s\n', warningFile);

end

function ensureOutputDirs(cfg)
dirs = {cfg.paths.tables, cfg.paths.logs, cfg.paths.predictions, cfg.paths.cache};
for k = 1:numel(dirs)
    if ~exist(dirs{k}, 'dir')
        mkdir(dirs{k});
    end
end
end

function validateFeatureColumns(T, featureNames)
missing = setdiff(featureNames, string(T.Properties.VariableNames), 'stable');
if ~isempty(missing)
    error('Observation table is missing required feature columns: %s', strjoin(missing, ', '));
end
end

function checkSubjectLeakage(trainTable, testTable, heldOutDomain)
overlap = intersect(unique(trainTable.subject_key), unique(testTable.subject_key));
if ~isempty(overlap)
    error('Subject leakage for held-out domain %s. Overlapping subject_key values: %s', ...
        heldOutDomain, strjoin(overlap(1:min(numel(overlap), 20)), ', '));
end
end

function info = describeDomainSplit(trainTable, testTable)
info = struct();
info.train_class1_subjects = countSubjects(trainTable, "class1");
info.train_class0_subjects = countSubjects(trainTable, "class0");
info.test_class1_subjects = countSubjects(testTable, "class1");
info.test_class0_subjects = countSubjects(testTable, "class0");
info.test_observations = height(testTable);
info.train_observations = height(trainTable);
info.test_classes = unique(testTable.analysis_label);
info.test_has_both_classes = numel(info.test_classes) == 2;
end

function [predTable, summaryRow, hyperRow] = runOneHeldOutDomain( ...
    trainTable, testTable, featureNames, heldOutDomain, trainingDomains, ...
    modelType, domainInfo, cfg, warningFid)

XTrainRaw = trainTable{:, cellstr(featureNames)};
XTestRaw = testTable{:, cellstr(featureNames)};
[XTrain, XTest, mu, sigma] = standardizeTrainTest(XTrainRaw, XTestRaw);
yTrain = categorical(trainTable.analysis_label);

bestParams = selectHyperparametersWithinTrainingDomains(trainTable, featureNames, modelType, cfg, warningFid);
model = fitSvmModel(XTrain, yTrain, modelType, bestParams);
[predicted, score] = predict(model, XTest);
positiveScore = extractPositiveScore(score, model.ClassNames, "class1");

predTable = makePredictionTable(testTable, predicted, positiveScore, heldOutDomain, trainingDomains, modelType);
metrics = computeMetrics(testTable.analysis_label, string(predicted), positiveScore);
[metrics, undefinedReason] = applyMetricAvailabilityRules(metrics, domainInfo);

summaryRow = makeSummaryRow(heldOutDomain, trainingDomains, modelType, domainInfo, metrics, undefinedReason);
hyperRow = makeHyperparameterRow(heldOutDomain, modelType, bestParams, mu, sigma);
end

function bestParams = selectHyperparametersWithinTrainingDomains(trainTable, featureNames, modelType, cfg, warningFid)
grid = makeSvmGrid(modelType, cfg);
requestedFolds = getInnerFoldCount(cfg);

try
    [foldId, ~, nFolds] = make_grouped_folds(trainTable.subject_key, trainTable.analysis_label, ...
        requestedFolds, cfg.RANDOM_SEED + 202);
catch ME
    logWarning(warningFid, 'Inner grouped validation failed for %s: %s. Using first grid value.', modelType, ME.message);
    bestParams = grid(1);
    return;
end

bestScore = -Inf;
bestParams = grid(1);
for g = 1:numel(grid)
    foldScores = NaN(nFolds, 1);
    for fold = 1:nFolds
        trainRows = foldId ~= fold;
        valRows = foldId == fold;

        if numel(unique(trainTable.analysis_label(trainRows))) < 2
            continue;
        end

        XTrainRaw = trainTable{trainRows, cellstr(featureNames)};
        XValRaw = trainTable{valRows, cellstr(featureNames)};
        [XTrain, XVal] = standardizeTrainTest(XTrainRaw, XValRaw);
        yTrain = categorical(trainTable.analysis_label(trainRows));

        try
            model = fitSvmModel(XTrain, yTrain, modelType, grid(g));
            predicted = predict(model, XVal);
            metrics = computeMetrics(trainTable.analysis_label(valRows), string(predicted), NaN(sum(valRows), 1));
            foldScores(fold) = metrics.balanced_accuracy;
        catch ME
            logWarning(warningFid, 'Inner validation failed for %s: %s', modelType, ME.message);
        end
    end

    score = mean(foldScores, 'omitnan');
    if isfinite(score) && score > bestScore
        bestScore = score;
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

function predTable = makePredictionTable(testTable, predicted, positiveScore, heldOutDomain, trainingDomains, modelType)
predTable = table( ...
    repmat(heldOutDomain, height(testTable), 1), ...
    repmat(trainingDomains, height(testTable), 1), ...
    repmat(modelType, height(testTable), 1), ...
    testTable.analysis_label, ...
    string(predicted), ...
    positiveScore, ...
    testTable.domain, ...
    testTable.time, ...
    testTable.subject_key, ...
    testTable.observation_key, ...
    'VariableNames', {'held_out_domain', 'training_domains', 'model_type', ...
    'analysis_label', 'predicted_label', 'positive_score', 'domain', 'time', ...
    'subject_key', 'observation_key'});
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

function [metrics, undefinedReason] = applyMetricAvailabilityRules(metrics, domainInfo)
undefinedReason = "";
if ~domainInfo.test_has_both_classes
    undefinedReason = "held-out domain contains only one class; balanced binary metrics are incomplete";
    metrics.balanced_accuracy = NaN;
    metrics.sensitivity = NaN;
    metrics.specificity = NaN;
    metrics.precision = NaN;
    metrics.f1 = NaN;
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

function row = makeSummaryRow(heldOutDomain, trainingDomains, modelType, info, metrics, undefinedReason)
row = table(heldOutDomain, trainingDomains, modelType, ...
    info.train_class1_subjects, info.train_class0_subjects, ...
    info.test_class1_subjects, info.test_class0_subjects, ...
    joinOrNone(info.test_classes), info.test_has_both_classes, ...
    metrics.accuracy, metrics.balanced_accuracy, metrics.sensitivity, ...
    metrics.specificity, metrics.precision, metrics.f1, metrics.roc_auc, ...
    metrics.tp, metrics.tn, metrics.fp, metrics.fn, string(undefinedReason), ...
    'VariableNames', {'held_out_domain', 'training_domains', 'model_type', ...
    'train_class1_subjects', 'train_class0_subjects', ...
    'test_class1_subjects', 'test_class0_subjects', ...
    'test_classes', 'test_has_both_classes', 'accuracy', 'balanced_accuracy', ...
    'sensitivity', 'specificity', 'precision', 'f1_score', 'roc_auc', ...
    'tp', 'tn', 'fp', 'fn', 'undefined_metric_reason'});
end

function row = makeHyperparameterRow(heldOutDomain, modelType, params, mu, sigma)
row = table(heldOutDomain, modelType, params.BoxConstraint, string(params.KernelScale), ...
    string(mat2str(mu, 4)), string(mat2str(sigma, 4)), ...
    'VariableNames', {'held_out_domain', 'model_type', 'BoxConstraint', ...
    'KernelScale', 'training_mean', 'training_sigma'});
end

function row = makeRuntimeRow(heldOutDomain, status, skipReason, runtimeSeconds)
row = table(heldOutDomain, status, string(skipReason), runtimeSeconds, ...
    'VariableNames', {'held_out_domain', 'status', 'skip_reason', 'runtime_seconds'});
end

function n = countSubjects(T, label)
rows = T.analysis_label == label;
if any(rows)
    n = numel(unique(T.subject_key(rows)));
else
    n = 0;
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

function writeOutputs(cfg, predictions, summaryTable, hyperparameterTable, runtimeTable)
writetable(predictions, fullfile(cfg.paths.predictions, 'leave_one_domain_predictions.csv'));
writetable(summaryTable, fullfile(cfg.paths.tables, 'leave_one_domain_summary.csv'));
writetable(hyperparameterTable, fullfile(cfg.paths.tables, 'leave_one_domain_selected_hyperparameters.csv'));
writetable(runtimeTable, fullfile(cfg.paths.tables, 'leave_one_domain_runtime.csv'));

save(fullfile(cfg.paths.cache, 'leave_one_domain_results.mat'), ...
    'predictions', 'summaryTable', 'hyperparameterTable', 'runtimeTable', '-v7.3');
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
