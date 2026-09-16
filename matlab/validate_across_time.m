function results = validate_across_time()
% Adapted from the original analysis implementation; identifiers and I/O generalized.
%VALIDATE_ACROSS_TIME Train at one TIME and test transfer to other TIME values.

cfg = config();
ensureOutputDirs(cfg);
rng(cfg.RANDOM_SEED);

warningFile = fullfile(cfg.paths.logs, 'validate_across_time_warnings.txt');
warningFid = fopen(warningFile, 'w');
if warningFid < 0
    error('Could not open warning log for writing: %s', warningFile);
end
cleanup = onCleanup(@() fclose(warningFid)); %#ok<NASGU>
fprintf(warningFid, 'Cross-TIME validation warnings\n');
fprintf(warningFid, 'Generated: %s\n\n', datestr(now, 31));

obsTable = load_cached_data("observation");
channels = cfg.expected_channels(:).';
featureNames = "mean_band_" + string(channels);
validateFeatureColumns(obsTable, featureNames);

models = getModelsToRun(cfg);
modes = "strict";
if runLongitudinalMode(cfg)
    modes = ["strict", "longitudinal"];
end

allResultParts = {};
allPredictionParts = {};

for modeIdx = 1:numel(modes)
    modeName = modes(modeIdx);
    fprintf('Cross-TIME validation mode: %s\n', modeName);
    for m = 1:numel(models)
        modelType = models(m);
        [resultTable, predictionTable] = runModeAndModel(obsTable, featureNames, cfg, ...
            modelType, modeName, warningFid);
        allResultParts{end + 1, 1} = resultTable; %#ok<AGROW>
        allPredictionParts{end + 1, 1} = predictionTable; %#ok<AGROW>
        saveHeatmaps(cfg, resultTable, modelType, modeName);
    end
end

resultsTable = vertcatOrEmpty(allResultParts);
predictionTable = vertcatOrEmpty(allPredictionParts);
writeOutputs(cfg, resultsTable, predictionTable);

results = struct();
results.results_table = resultsTable;
results.predictions = predictionTable;
results.warning_file = warningFile;

fprintf('Cross-TIME validation complete.\n');
fprintf('Result rows: %d\n', height(resultsTable));
fprintf('Prediction rows: %d\n', height(predictionTable));
fprintf('Outputs saved to: %s\n', cfg.paths.tables);
fprintf('Warnings saved to: %s\n', warningFile);

end

function ensureOutputDirs(cfg)
dirs = {cfg.paths.tables, cfg.paths.logs, cfg.paths.predictions, cfg.paths.figures, cfg.paths.cache};
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

function models = getModelsToRun(cfg)
models = "linear_svm";
runRbf = true;
if isfield(cfg, 'RUN_RBF_CROSS_TIME')
    runRbf = logical(cfg.RUN_RBF_CROSS_TIME);
end
if runRbf
    models = ["linear_svm", "rbf_svm"];
end
end

function tf = runLongitudinalMode(cfg)
tf = false;
if isfield(cfg, 'RUN_LONGITUDINAL_TIME_TRANSFER')
    tf = logical(cfg.RUN_LONGITUDINAL_TIME_TRANSFER);
elseif isfield(cfg, 'analysis') && isfield(cfg.analysis, 'run_longitudinal_time_transfer')
    tf = logical(cfg.analysis.run_longitudinal_time_transfer);
end
end

function [resultTable, predictionTable] = runModeAndModel(obsTable, featureNames, cfg, modelType, modeName, warningFid)
timeValues = sort(unique(obsTable.time)).';
resultParts = {};
predictionParts = {};

for trainIdx = 1:numel(timeValues)
    trainTime = timeValues(trainIdx);
    trainRows = obsTable.time == trainTime;
    trainTable = obsTable(trainRows, :);
    tTrain = tic;

    [canTrain, trainSkipReason] = validateTrainTime(trainTable);
    if ~canTrain
        logWarning(warningFid, 'Skipping train TIME %s for %s/%s: %s', ...
            string(trainTime), modelType, modeName, trainSkipReason);
        resultParts{end + 1, 1} = skippedRowsForTrainTime(timeValues, trainTime, modelType, modeName, trainSkipReason); %#ok<AGROW>
        continue;
    end

    bestParams = selectHyperparametersWithinTrainTime(trainTable, featureNames, modelType, cfg, warningFid);
    XTrainRaw = trainTable{:, cellstr(featureNames)};
    [XTrain, ~, mu, sigma] = standardizeTrainTest(XTrainRaw, XTrainRaw);
    yTrain = categorical(trainTable.analysis_label);
    model = fitSvmModel(XTrain, yTrain, modelType, bestParams);

    for testIdx = 1:numel(timeValues)
        testTime = timeValues(testIdx);
        tCombo = tic;
        testTableOriginal = obsTable(obsTable.time == testTime, :);
        [testTable, removedSubjectCount] = applyTransferMode(testTableOriginal, trainTable, modeName);
        overlapSubjects = intersect(unique(trainTable.subject_key), unique(testTable.subject_key));

        warnOnTimeDomainMismatch(trainTable, testTableOriginal, trainTime, testTime, modeName, warningFid);
        if modeName == "strict" && removedSubjectCount > 0
            logWarning(warningFid, 'Strict mode train TIME %s -> test TIME %s removed %d test subjects seen in training TIME.', ...
                string(trainTime), string(testTime), removedSubjectCount);
        elseif modeName == "longitudinal" && ~isempty(overlapSubjects)
            logWarning(warningFid, 'Longitudinal mode train TIME %s -> test TIME %s includes %d repeated subjects across TIME.', ...
                string(trainTime), string(testTime), numel(overlapSubjects));
        end

        [canTest, testSkipReason, testInfo] = validateTestTime(testTable);
        if ~canTest
            resultParts{end + 1, 1} = makeResultRow(trainTime, testTime, modelType, modeName, ...
                bestParams, testInfo, "skipped", testSkipReason, toc(tCombo), emptyMetrics()); %#ok<AGROW>
            continue;
        end

        XTest = (testTable{:, cellstr(featureNames)} - mu) ./ sigma;
        [predicted, score] = predict(model, XTest);
        positiveScore = extractPositiveScore(score, model.ClassNames, "class1");
        metrics = computeMetrics(testTable.analysis_label, string(predicted), positiveScore);

        resultParts{end + 1, 1} = makeResultRow(trainTime, testTime, modelType, modeName, ...
            bestParams, testInfo, "completed", "", toc(tCombo), metrics); %#ok<AGROW>
        predictionParts{end + 1, 1} = makePredictionTable(testTable, predicted, positiveScore, ...
            trainTime, testTime, modelType, modeName, bestParams); %#ok<AGROW>
    end

    fprintf('  %s %s train TIME %s done in %.2f seconds\n', modeName, modelType, string(trainTime), toc(tTrain));
end

resultTable = vertcatOrEmpty(resultParts);
predictionTable = vertcatOrEmpty(predictionParts);
end

function [canTrain, reason] = validateTrainTime(T)
canTrain = true;
reason = "";
if height(T) < 6
    canTrain = false;
    reason = "fewer than 6 training observations";
elseif numel(unique(T.analysis_label)) < 2
    canTrain = false;
    reason = "training TIME has only one class";
elseif countSubjects(T, "class1") < 2
    canTrain = false;
    reason = "training TIME has fewer than 2 class1 subjects";
elseif countSubjects(T, "class0") < 2
    canTrain = false;
    reason = "training TIME has fewer than 2 class0 subjects";
end
end

function [canTest, reason, info] = validateTestTime(T)
info = describeTestTime(T);
canTest = true;
reason = "";
if height(T) < 2
    canTest = false;
    reason = "fewer than 2 test observations after filtering";
elseif numel(unique(T.analysis_label)) < 2
    canTest = false;
    reason = "test TIME has only one class after filtering";
elseif info.class1_test_subjects < 1
    canTest = false;
    reason = "test TIME has no class1 subjects after filtering";
elseif info.class0_test_subjects < 1
    canTest = false;
    reason = "test TIME has no class0 subjects after filtering";
end
end

function info = describeTestTime(T)
info = struct();
info.test_observations = height(T);
if isempty(T)
    info.total_test_subjects = 0;
    info.class1_test_subjects = 0;
    info.class0_test_subjects = 0;
    info.test_domains = "<none>";
else
    info.total_test_subjects = numel(unique(T.subject_key));
    info.class1_test_subjects = countSubjects(T, "class1");
    info.class0_test_subjects = countSubjects(T, "class0");
    info.test_domains = joinOrNone(unique(T.domain));
end
end

function [testTable, removedSubjectCount] = applyTransferMode(testTableOriginal, trainTable, modeName)
removedSubjectCount = 0;
testTable = testTableOriginal;
if modeName == "strict"
    trainSubjects = unique(trainTable.subject_key);
    removeRows = ismember(testTable.subject_key, trainSubjects);
    removedSubjectCount = numel(unique(testTable.subject_key(removeRows)));
    testTable = testTable(~removeRows, :);
end
end

function warnOnTimeDomainMismatch(trainTable, testTable, trainTime, testTime, modeName, warningFid)
trainDomains = unique(trainTable.domain);
testDomains = unique(testTable.domain);
if ~isempty(trainDomains) && ~isempty(testDomains) && isempty(intersect(trainDomains, testDomains))
    logWarning(warningFid, '%s mode train TIME %s -> test TIME %s uses different domain sets. Train [%s], test [%s].', ...
        modeName, string(trainTime), string(testTime), joinOrNone(trainDomains), joinOrNone(testDomains));
end
end

function bestParams = selectHyperparametersWithinTrainTime(trainTable, featureNames, modelType, cfg, warningFid)
grid = makeSvmGrid(modelType, cfg);
requestedFolds = getInnerFoldCount(cfg);

try
    [foldId, ~, nFolds] = make_grouped_folds(trainTable.subject_key, trainTable.analysis_label, ...
        requestedFolds, cfg.RANDOM_SEED + round(trainTable.time(1)));
catch ME
    logWarning(warningFid, 'Inner grouped validation failed for train TIME %s, %s: %s. Using first grid value.', ...
        string(trainTable.time(1)), modelType, ME.message);
    bestParams = grid(1);
    return;
end

bestScore = -Inf;
bestParams = grid(1);
for g = 1:numel(grid)
    scores = NaN(nFolds, 1);
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
            scores(fold) = metrics.balanced_accuracy;
        catch ME
            logWarning(warningFid, 'Inner validation failed for train TIME %s, %s: %s', ...
                string(trainTable.time(1)), modelType, ME.message);
        end
    end

    score = mean(scores, 'omitnan');
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

function metrics = emptyMetrics()
metrics = struct('accuracy', NaN, 'balanced_accuracy', NaN, 'sensitivity', NaN, ...
    'specificity', NaN, 'precision', NaN, 'f1', NaN, 'roc_auc', NaN);
end

function value = safeDivide(a, b)
if b == 0
    value = NaN;
else
    value = a / b;
end
end

function row = makeResultRow(trainTime, testTime, modelType, modeName, params, testInfo, status, skipReason, runtimeSeconds, metrics)
row = table(trainTime, testTime, modelType, modeName, status, string(skipReason), ...
    testInfo.test_observations, testInfo.total_test_subjects, ...
    testInfo.class1_test_subjects, testInfo.class0_test_subjects, ...
    string(testInfo.test_domains), params.BoxConstraint, string(params.KernelScale), ...
    metrics.accuracy, metrics.balanced_accuracy, metrics.f1, metrics.roc_auc, ...
    metrics.sensitivity, metrics.specificity, metrics.precision, runtimeSeconds, ...
    'VariableNames', {'train_time', 'test_time', 'model_type', 'transfer_mode', ...
    'status', 'skip_reason', 'test_observations', 'total_test_subjects', ...
    'class1_test_subjects', 'class0_test_subjects', 'test_domains', ...
    'BoxConstraint', 'KernelScale', 'accuracy', 'balanced_accuracy', 'f1_score', ...
    'roc_auc', 'sensitivity', 'specificity', 'precision', 'runtime_seconds'});
end

function rows = skippedRowsForTrainTime(timeValues, trainTime, modelType, modeName, skipReason)
parts = cell(numel(timeValues), 1);
for i = 1:numel(timeValues)
    testInfo = struct('test_observations', 0, 'total_test_subjects', 0, ...
        'class1_test_subjects', 0, 'class0_test_subjects', 0, 'test_domains', "<none>");
    params = struct('BoxConstraint', NaN, 'KernelScale', "none");
    parts{i} = makeResultRow(trainTime, timeValues(i), modelType, modeName, params, ...
        testInfo, "skipped", skipReason, 0, emptyMetrics());
end
rows = vertcat(parts{:});
end

function predictionTable = makePredictionTable(testTable, predicted, positiveScore, trainTime, testTime, modelType, modeName, params)
predictionTable = table(repmat(trainTime, height(testTable), 1), repmat(testTime, height(testTable), 1), ...
    repmat(modelType, height(testTable), 1), repmat(modeName, height(testTable), 1), ...
    testTable.analysis_label, string(predicted), positiveScore, testTable.domain, ...
    testTable.subject_key, testTable.observation_key, repmat(params.BoxConstraint, height(testTable), 1), ...
    repmat(string(params.KernelScale), height(testTable), 1), ...
    'VariableNames', {'train_time', 'test_time', 'model_type', 'transfer_mode', ...
    'analysis_label', 'predicted_label', 'positive_score', 'domain', ...
    'subject_key', 'observation_key', 'BoxConstraint', 'KernelScale'});
end

function saveHeatmaps(cfg, resultTable, modelType, modeName)
if isempty(resultTable)
    return;
end
timeValues = sort(unique([resultTable.train_time; resultTable.test_time])).';
metrics = ["balanced_accuracy", "f1_score", "roc_auc", "class1_test_subjects", "class0_test_subjects"];
for metricIdx = 1:numel(metrics)
    metricName = metrics(metricIdx);
    matrix = NaN(numel(timeValues), numel(timeValues));
    for i = 1:numel(timeValues)
        for j = 1:numel(timeValues)
            rows = resultTable.train_time == timeValues(i) & resultTable.test_time == timeValues(j) & resultTable.status == "completed";
            if any(rows)
                matrix(i, j) = resultTable.(metricName)(find(rows, 1));
            end
        end
    end
    fig = figure('Visible', 'off', 'Position', [100, 100, 800, 700]);
    imagesc(timeValues, timeValues, matrix);
    set(gca, 'YDir', 'normal');
    colorbar;
    xlabel('Test TIME');
    ylabel('Train TIME');
    title(sprintf('%s %s: %s', modeName, modelType, strrep(metricName, '_', ' ')));
    saveas(fig, fullfile(cfg.paths.figures, sprintf('cross_time_%s_%s_%s_heatmap.png', modeName, modelType, metricName)));
    close(fig);
end
end

function writeOutputs(cfg, resultsTable, predictionTable)
writetable(resultsTable, fullfile(cfg.paths.tables, 'cross_time_transfer_results.csv'));
writetable(predictionTable, fullfile(cfg.paths.predictions, 'cross_time_transfer_predictions.csv'));
save(fullfile(cfg.paths.cache, 'cross_time_transfer_results.mat'), ...
    'resultsTable', 'predictionTable', '-v7.3');
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
