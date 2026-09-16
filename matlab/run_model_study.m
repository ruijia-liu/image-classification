function results = run_model_study(options)
%RUN_MODEL_STUDY Adapted model comparison, transfer, importance and top-k study.
% Uses private cached features; all generated files stay in the private workspace.
if nargin < 1, options = struct(); end
base = config();
cfg = struct('RandomSeed',42, 'NumFolds',3, 'InnerFolds',2, ...
    'NumPermutations',3, 'TopKValues',[1 2 numel(base.expected_channels)]);
cfg.Models = ["rbf_svm","linear_svm","logistic","random_forest","boosted_trees"];
cfg.CrossExcludeDomains = strings(0,1);
allowed = ["Models","NumFolds","InnerFolds","NumPermutations","TopKValues","RandomSeed"];
for name = string(fieldnames(options))'
    assert(ismember(name,allowed), 'Unsupported study option.');
    cfg.(name) = options.(name);
end
cfg.PositiveClass = "class1"; cfg.NegativeClass = "class0";
cfg.BandNames = "band_" + string(base.expected_channels);
cfg.Channels = base.expected_channels;
T = load_cached_data("patch");
T.LABEL = categorical(T.analysis_label, ["class0","class1"]);
T.group_id = T.subject_key;
T.analysis_set = T.domain + "_time_" + string(T.time);
assert(~isempty(build_sets(T)), 'No subset has enough groups per class.');
folder = fullfile(base.private_root, 'model_study');
assert(~isfolder(folder), 'Use a fresh private workspace for each model study.');
mkdir(folder);
rng(cfg.RandomSeed);
results = run_dataset_analysis(T, cfg, folder, "image_features");
end

function results = run_dataset_analysis(T, cfg, outDir, datasetName)
writetable(summarise_dataset(T), fullfile(outDir, 'dataset_summary.csv'));
sets = build_sets(T);
allBest = table();
allModel = table();
allFold = table();
allImp = table();
allTopK = table();

for s = 1:numel(sets)
    name = string(sets(s).Name);
    Ts = sets(s).Table;
    fprintf('\nWITHIN %s (%s): %d rows, %d groups\n', name, datasetName, height(Ts), numel(unique(Ts.group_id)));
    setDir = fullfile(outDir, 'within', sanitize_filename(name));
    if ~exist(setDir, 'dir'), mkdir(setDir); end

    [summary, folds] = compare_models_within(Ts, cfg);
    summary.dataset = repmat(datasetName, height(summary), 1);
    summary.analysis_set = repmat(name, height(summary), 1);
    folds.dataset = repmat(datasetName, height(folds), 1);
    folds.analysis_set = repmat(name, height(folds), 1);
    summary = movevars(summary, {'dataset','analysis_set'}, 'Before', 1);
    folds = movevars(folds, {'dataset','analysis_set'}, 'Before', 1);
    writetable(summary, fullfile(setDir, 'model_comparison.csv'));
    writetable(folds, fullfile(setDir, 'model_fold_metrics.csv'));

    best = summary(1, :);
    writetable(best, fullfile(setDir, 'best_model.csv'));
    topModels = summary(1:min(2, height(summary)), :);
    writetable(topModels, fullfile(setDir, 'top2_models.csv'));
    top3Models = summary(1:min(3, height(summary)), :);
    writetable(top3Models, fullfile(setDir, 'top3_models.csv'));
    imp = permutation_importance(Ts, best.model, parse_params(best.hyperparameters), cfg);
    imp.dataset = repmat(datasetName, height(imp), 1);
    imp.analysis_set = repmat(name, height(imp), 1);
    imp.domain = repmat(Ts.domain(1), height(imp), 1);
    imp.time = repmat(Ts.time(1), height(imp), 1);
    imp.model = repmat(string(best.model), height(imp), 1);
    imp.hyperparameters = repmat(string(best.hyperparameters), height(imp), 1);
    imp = movevars(imp, {'dataset','analysis_set','domain','time','model','hyperparameters'}, 'Before', 1);
    writetable(imp, fullfile(setDir, 'band_importance.csv'));

    topK = topk_performance(Ts, best.model, parse_params(best.hyperparameters), imp, cfg);
    topK.dataset = repmat(datasetName, height(topK), 1);
    topK.analysis_set = repmat(name, height(topK), 1);
    topK.domain = repmat(Ts.domain(1), height(topK), 1);
    topK.time = repmat(Ts.time(1), height(topK), 1);
    topK.model = repmat("selected_inside_each_outer_fold", height(topK), 1);
    topK.hyperparameters = repmat("tuned_inside_each_outer_fold", height(topK), 1);
    topK = movevars(topK, {'dataset','analysis_set','domain','time','model','hyperparameters'}, 'Before', 1);
    writetable(topK, fullfile(setDir, 'topK_performance.csv'));

    plot_spectral(Ts, cfg, setDir, name);
    plot_importance(imp, cfg, setDir, name);
    plot_topk(topK, setDir, name);
    plot_model_ranking(summary, setDir, name);

    allBest = [allBest; best]; %#ok<AGROW>
    allModel = [allModel; summary]; %#ok<AGROW>
    allFold = [allFold; folds]; %#ok<AGROW>
    allImp = [allImp; imp]; %#ok<AGROW>
    allTopK = [allTopK; topK]; %#ok<AGROW>
end

writetable(allBest, fullfile(outDir, 'ALL_within_best_models.csv'));
writetable(allModel, fullfile(outDir, 'ALL_within_model_comparison.csv'));
writetable(allFold, fullfile(outDir, 'ALL_within_fold_metrics.csv'));
writetable(allImp, fullfile(outDir, 'ALL_within_band_importance.csv'));
writetable(allTopK, fullfile(outDir, 'ALL_within_topK_performance.csv'));
if ~isempty(allModel)
    allTop2 = allModel(allModel.selection_rank <= 2, :);
    writetable(allTop2, fullfile(outDir, 'ALL_within_top2_models.csv'));
    allTop3 = allModel(allModel.selection_rank <= 3, :);
    writetable(allTop3, fullfile(outDir, 'ALL_within_top3_models.csv'));
end
plot_spectral_by_domain(sets, cfg, outDir, datasetName);

crossDir = fullfile(outDir, 'cross_tests');
if ~exist(crossDir, 'dir'), mkdir(crossDir); end
cross = run_cross_tests(T, cfg, crossDir, datasetName, allModel);

results = struct();
results.WithinBest = allBest;
results.WithinImportance = allImp;
results.WithinTopK = allTopK;
results.Cross = cross;
end

function sets = build_sets(T)
names = unique(T.analysis_set, 'stable');
sets = struct('Name', {}, 'Table', {});
for i = 1:numel(names)
    idx = T.analysis_set == names(i);
    if min_class_group_count(T.group_id(idx), T.LABEL(idx)) >= 4
        sets(end+1).Name = names(i); %#ok<AGROW>
        sets(end).Table = T(idx, :);
    end
end
end

function [summary, folds] = compare_models_within(T, cfg)
models = string(cfg.Models);
foldId = make_grouped_folds(T.group_id, T.LABEL, cfg.NumFolds, cfg.RandomSeed);
folds = table();
for m = 1:numel(models)
    model = models(m);
    for f = 1:max(foldId)
        trainIdx = foldId ~= f;
        testIdx = foldId == f;
        tuned = tune_model(T(trainIdx, :), model, cfg);
        fit = train_model(T{trainIdx, cfg.BandNames}, T.LABEL(trainIdx), model, tuned.Params);
        [yp, sc] = predict_model(fit, T{testIdx, cfg.BandNames}, cfg);
        pm = classification_metrics(T.LABEL(testIdx), yp, sc, cfg);
        lm = aggregate_metrics(T(testIdx, :), yp, sc, cfg, "object");
        sourcem = aggregate_metrics(T(testIdx, :), yp, sc, cfg, "source");
        gm = aggregate_metrics(T(testIdx, :), yp, sc, cfg, "group");
        folds = [folds; metric_row(pm, model, tuned.Hyperparameters, f, "patch", sum(testIdx), numel(unique(T.group_id(testIdx))))]; %#ok<AGROW>
        folds = [folds; metric_row(lm, model, tuned.Hyperparameters, f, "object", sum(testIdx), numel(unique(make_aggregate_ids(T(testIdx, :), "object"))))]; %#ok<AGROW>
        folds = [folds; metric_row(sourcem, model, tuned.Hyperparameters, f, "source", sum(testIdx), numel(unique(make_aggregate_ids(T(testIdx, :), "source"))))]; %#ok<AGROW>
        folds = [folds; metric_row(gm, model, tuned.Hyperparameters, f, "group", sum(testIdx), numel(unique(T.group_id(testIdx))))]; %#ok<AGROW>
    end
end
summary = summarise_model_folds(folds);
end

function row = metric_row(metrics, model, hyper, fold, level, nRows, nGroups)
row = struct2table(metrics);
row.model = string(model);
row.hyperparameters = string(hyper);
row.fold = fold;
row.level = string(level);
row.n_test_rows = nRows;
row.n_test_groups = nGroups;
row = movevars(row, {'model','hyperparameters','level','fold','n_test_rows','n_test_groups'}, 'Before', 1);
end

function summary = summarise_model_folds(folds)
scoreLevels = ["source","object","group"];
scoreRows = folds(ismember(folds.level, scoreLevels), :);
patchRows = folds(folds.level == "patch", :);
groupRows = folds(folds.level == "group", :);
objectRows = folds(folds.level == "object", :);
sourceRows = folds(folds.level == "source", :);
keys = unique(groupRows(:, {'model'}), 'rows', 'stable');
summary = table();
for i = 1:height(keys)
    rows = groupRows(groupRows.model == keys.model(i), :);
    lrows = objectRows(objectRows.model == keys.model(i), :);
    prows = sourceRows(sourceRows.model == keys.model(i), :);
    patchrows = patchRows(patchRows.model == keys.model(i), :);
    srows = scoreRows(scoreRows.model == keys.model(i), :);
    row = table();
    row.model = keys.model(i);
    row.hyperparameters = mode_string(srows.hyperparameters);
    row.selection_accuracy_mean = mean([mean(prows.accuracy, 'omitnan') mean(lrows.accuracy, 'omitnan') mean(rows.accuracy, 'omitnan')], 'omitnan');
    row.patch_accuracy_mean = mean(patchrows.accuracy, 'omitnan');
    row.patch_accuracy_std = std(patchrows.accuracy, 'omitnan');
    row.source_accuracy_mean = mean(prows.accuracy, 'omitnan');
    row.source_accuracy_std = std(prows.accuracy, 'omitnan');
    row.object_accuracy_mean = mean(lrows.accuracy, 'omitnan');
    row.object_accuracy_std = std(lrows.accuracy, 'omitnan');
    row.group_accuracy_mean = mean(rows.accuracy, 'omitnan');
    row.group_accuracy_std = std(rows.accuracy, 'omitnan');
    row.group_f1_mean = mean(rows.f1_positive, 'omitnan');
    row.group_f1_std = std(rows.f1_positive, 'omitnan');
    row.model_complexity_rank = model_complexity_rank(keys.model(i));
    summary = [summary; row]; %#ok<AGROW>
end
summary = sortrows(summary, {'selection_accuracy_mean','source_accuracy_mean','object_accuracy_mean','group_accuracy_mean','model_complexity_rank'}, {'descend','descend','descend','descend','ascend'});
summary.selection_rank = (1:height(summary))';
summary.is_selected_best_model = summary.selection_rank == 1;
summary.is_second_model = summary.selection_rank == 2;
summary = movevars(summary, {'selection_rank','is_selected_best_model'}, 'Before', 1);
end

function cross = run_cross_tests(T, cfg, outDir, datasetName, withinModelSelection)
fprintf('\nCROSS tests for %s\n', datasetName);
Tc = T(~ismember(T.domain, cfg.CrossExcludeDomains), :);
writetable(summarise_dataset(Tc), fullfile(outDir, 'cross_dataset_summary.csv'));
sets = build_sets(Tc);
rows = table();
selectionRows = table();

for i = 1:numel(sets)
    trainSet = sets(i);
    for j = 1:numel(sets)
        if i == j, continue; end
        testSet = sets(j);
        mode = "all_pairwise";
        if trainSet.Table.time(1) == testSet.Table.time(1)
            mode = "same_time_cross_domain";
        elseif trainSet.Table.domain(1) == testSet.Table.domain(1)
            mode = "within_domain_cross_time";
        end
        
        trainSelection = withinModelSelection(withinModelSelection.analysis_set == string(trainSet.Name), :);
        best = trainSelection(1, :);
        trainSelection.dataset = repmat(datasetName, height(trainSelection), 1);
        trainSelection.cross_mode = repmat(mode, height(trainSelection), 1);
        trainSelection.train_set = repmat(string(trainSet.Name), height(trainSelection), 1);
        trainSelection.test_set = repmat(string(testSet.Name), height(trainSelection), 1);
        trainSelection = movevars(trainSelection, {'dataset','cross_mode','train_set','test_set'}, 'Before', 1);
        selectionRows = [selectionRows; trainSelection]; %#ok<AGROW>

        testTable = testSet.Table(~ismember(testSet.Table.group_id, trainSet.Table.group_id), :);
        if isempty(testTable) || numel(unique(testTable.LABEL)) < 2
            warning('Skipping transfer with insufficient unseen sources.');
            continue;
        end
        tuned = tune_model(trainSet.Table, best.model(1), cfg);
        row = evaluate_train_test(trainSet.Table, testTable, best.model(1), tuned.Params, cfg);
        row.dataset = datasetName;
        row.cross_mode = mode;
        row.train_set = string(trainSet.Name);
        row.test_set = string(testSet.Name);
        row.model = string(best.model(1));
        row.hyperparameters = string(tuned.Hyperparameters);
        row = movevars(row, {'dataset','cross_mode','train_set','test_set','model','hyperparameters'}, 'Before', 1);
        rows = [rows; row]; %#ok<AGROW>
    end
end

if ~isempty(rows)
    writetable(rows, fullfile(outDir, 'cross_pairwise_results.csv'));
    writetable(selectionRows, fullfile(outDir, 'cross_training_side_model_selection.csv'));
    plot_cross_heatmap(rows, outDir, datasetName);
end
cross.Pairwise = rows;
cross.TrainingSideSelection = selectionRows;
end

function row = evaluate_train_test(trainT, testT, model, params, cfg)
assert(isempty(intersect(trainT.group_id, testT.group_id)), 'Source overlap in transfer evaluation.');
fit = train_model(trainT{:, cfg.BandNames}, trainT.LABEL, model, params);
[yp, sc] = predict_model(fit, testT{:, cfg.BandNames}, cfg);
pm = classification_metrics(testT.LABEL, yp, sc, cfg);
objectm = aggregate_metrics(testT, yp, sc, cfg, "object");
sourcem = aggregate_metrics(testT, yp, sc, cfg, "source");
gm = aggregate_metrics(testT, yp, sc, cfg, "group");
row = table();
row.patch_balanced_accuracy = pm.balanced_accuracy;
row.patch_f1 = pm.f1_positive;
row.patch_accuracy = pm.accuracy;
row.selection_accuracy = mean([sourcem.accuracy objectm.accuracy gm.accuracy], 'omitnan');
row.source_accuracy = sourcem.accuracy;
row.object_accuracy = objectm.accuracy;
row.group_accuracy = gm.accuracy;
row.group_f1 = gm.f1_positive;
row.n_train_groups = numel(unique(trainT.group_id));
row.n_test_groups = numel(unique(testT.group_id));
row.n_train_rows = height(trainT);
row.n_test_rows = height(testT);
end

function tuned = tune_model(T, modelName, cfg)
grid = tuning_grid(modelName);
if numel(grid) == 1 || min_class_group_count(T.group_id, T.LABEL) < 2
    tuned = grid(1);
    return;
end
foldId = make_grouped_folds(T.group_id, T.LABEL, min(cfg.InnerFolds, min_class_group_count(T.group_id, T.LABEL)), cfg.RandomSeed + 17);
score = NaN(numel(grid), max(foldId));
for g = 1:numel(grid)
    for f = 1:max(foldId)
        tr = foldId ~= f; va = foldId == f;
        fit = train_model(T{tr, cfg.BandNames}, T.LABEL(tr), modelName, grid(g).Params);
        [yp, sc] = predict_model(fit, T{va, cfg.BandNames}, cfg);
        score(g, f) = selection_accuracy(T(va, :), yp, sc, cfg);
    end
end
[~, idx] = max(mean(score, 2, 'omitnan'));
tuned = grid(idx);
end

function grid = tuning_grid(modelName)
grid = struct('Params', {}, 'Hyperparameters', {});
switch string(modelName)
    case "rbf_svm"
        cVals = [0.1 1 10];
        kernelScales = [0.5 1 2 4];
        for c = cVals
            for ks = kernelScales
                grid(end+1) = mkcfg(struct('BoxConstraint', c, 'KernelScale', ks)); %#ok<AGROW>
            end
        end
    case "linear_svm"
        for c = [0.1 1 10]
            grid(end+1) = mkcfg(struct('BoxConstraint', c)); %#ok<AGROW>
        end
    case "logistic"
        for l = [1e-4 1e-3 1e-2 1e-1]
            grid(end+1) = mkcfg(struct('Lambda', l)); %#ok<AGROW>
        end
    case "random_forest"
        for object = [1 5]
            grid(end+1) = mkcfg(struct('NumTrees', 80, 'MinLeafSize', object)); %#ok<AGROW>
        end
    case "boosted_trees"
        grid(end+1) = mkcfg(struct('NumLearningCycles', 40, 'LearnRate', 0.1, 'MinLeafSize', 5, 'MaxNumSplits', 6)); %#ok<AGROW>
end
end

function c = mkcfg(params)
c.Params = params;
c.Hyperparameters = params_to_string(params);
end

function imp = permutation_importance(T, model, params, cfg)
foldId = make_grouped_folds(T.group_id, T.LABEL, cfg.NumFolds, cfg.RandomSeed + 99);
deltaSelection = NaN(max(foldId), numel(cfg.BandNames), cfg.NumPermutations);
deltaF1 = deltaSelection;
for f = 1:max(foldId)
    tr = foldId ~= f; te = foldId == f;
    fit = train_model(T{tr, cfg.BandNames}, T.LABEL(tr), model, params);
    [yp, sc] = predict_model(fit, T{te, cfg.BandNames}, cfg);
    baseSelection = selection_accuracy(T(te, :), yp, sc, cfg);
    baseGroup = aggregate_metrics(T(te, :), yp, sc, cfg, "group");
    X = T{te, cfg.BandNames};
    for b = 1:numel(cfg.BandNames)
        for p = 1:cfg.NumPermutations
            Xp = X; Xp(:, b) = Xp(randperm(size(Xp, 1)), b);
            [yp2, sc2] = predict_model(fit, Xp, cfg);
            mmSelection = selection_accuracy(T(te, :), yp2, sc2, cfg);
            mmGroup = aggregate_metrics(T(te, :), yp2, sc2, cfg, "group");
            deltaSelection(f, b, p) = baseSelection - mmSelection;
            deltaF1(f, b, p) = baseGroup.f1_positive - mmGroup.f1_positive;
        end
    end
end
flatSelection = reshape(permute(deltaSelection, [1 3 2]), [], numel(cfg.BandNames));
flatF1 = reshape(permute(deltaF1, [1 3 2]), [], numel(cfg.BandNames));
imp = table(cfg.BandNames(:), cfg.Channels(:), mean(flatSelection, 1, 'omitnan')', std(flatSelection, 0, 1, 'omitnan')', mean(flatF1, 1, 'omitnan')', std(flatF1, 0, 1, 'omitnan')', ...
    'VariableNames', {'band','channel','permutation_delta_selection_accuracy_mean','permutation_delta_selection_accuracy_std','permutation_delta_f1_mean','permutation_delta_f1_std'});
imp.rank_selection_accuracy = tiedrank_desc(imp.permutation_delta_selection_accuracy_mean);
imp.rank_f1 = tiedrank_desc(imp.permutation_delta_f1_mean);
imp.mean_rank = mean([imp.rank_selection_accuracy imp.rank_f1], 2, 'omitnan');
imp = sortrows(imp, 'mean_rank', 'ascend');
end

function topK = topk_performance(T, ~, ~, ~, cfg)
% Model family, feature ranking and tuning all use the outer training rows.
foldId = make_grouped_folds(T.group_id, T.LABEL, cfg.NumFolds, cfg.RandomSeed + 211);
rows = {};
for f = 1:max(foldId)
    trainT = T(foldId ~= f,:); testT = T(foldId == f,:);
    [comparison, ~] = compare_models_within(trainT, cfg);
    model = comparison.model(1);
    tuned = tune_model(trainT, model, cfg);
    imp = permutation_importance(trainT, model, tuned.Params, cfg);
    ranked = string(imp.band);
    for k = unique(min(cfg.TopKValues, numel(ranked)))
        local = cfg; local.BandNames = ranked(1:k)';
        chosen = tune_model(trainT, model, local);
        fit = train_model(trainT{:,local.BandNames}, trainT.LABEL, model, chosen.Params);
        [yp, sc] = predict_model(fit, testT{:,local.BandNames}, local);
        score = selection_accuracy(testT, yp, sc, local);
        gm = aggregate_metrics(testT, yp, sc, local, "group");
        rows{end+1,1} = table(k, f, strjoin(local.BandNames, ','), score, gm.f1_positive, ...
            'VariableNames', {'k','fold','bands','selection_accuracy','group_f1'}); %#ok<AGROW>
    end
end
folds = vertcat(rows{:});
topK = table();
for k = unique(folds.k)'
    part = folds(folds.k == k,:);
    row = table(k, strjoin(unique(part.bands), ' | '), mean(part.selection_accuracy), ...
        std(part.selection_accuracy), mean(part.group_f1,'omitnan'), std(part.group_f1,'omitnan'), ...
        'VariableNames', {'k','bands_by_fold','selection_accuracy_mean','selection_accuracy_std', ...
        'group_f1_mean','group_f1_std'});
    topK = [topK; row]; %#ok<AGROW>
end
end

function fit = train_model(X, y, modelName, params)
fit = struct('model', string(modelName), 'scaler', []);
if any(string(modelName) == ["rbf_svm","linear_svm","logistic"])
    [X, fit.scaler] = robust_scale_fit(X);
end
switch string(modelName)
    case "rbf_svm"
        fit.obj = fitcsvm(X, y, 'KernelFunction','rbf','KernelScale',params.KernelScale,'BoxConstraint',params.BoxConstraint,'Standardize',false,'ClassNames',categories(y));
    case "linear_svm"
        fit.obj = fitcsvm(X, y, 'KernelFunction','linear','BoxConstraint',params.BoxConstraint,'Standardize',false,'ClassNames',categories(y));
    case "logistic"
        fit.obj = fitclinear(X, y, 'Learner','logistic','Regularization','ridge','Lambda',params.Lambda,'ClassNames',categories(y));
    case "random_forest"
        fit.obj = TreeBagger(params.NumTrees, X, cellstr(y), 'Method','classification','MinLeafSize',params.MinLeafSize,'NumPredictorsToSample',max(1,floor(sqrt(size(X,2)))));
    case "boosted_trees"
        t = templateTree('MinLeafSize', params.MinLeafSize, 'MaxNumSplits', params.MaxNumSplits);
        fit.obj = fitcensemble(X, y, 'Method','AdaBoostM1','Learners',t,'NumLearningCycles',params.NumLearningCycles,'LearnRate',params.LearnRate,'ClassNames',categories(y));
end
end

function [yp, score] = predict_model(fit, X, cfg)
if ~isempty(fit.scaler), X = robust_scale_apply(X, fit.scaler); end
switch fit.model
    case "random_forest"
        [pc, sc] = predict(fit.obj, X);
        yp = categorical(string(pc), categories(categorical([cfg.NegativeClass cfg.PositiveClass])));
        cls = string(fit.obj.ClassNames);
    otherwise
        [yp, sc] = predict(fit.obj, X);
        cls = string(fit.obj.ClassNames);
end
pos = find(cls == cfg.PositiveClass, 1);
if isempty(pos), pos = size(sc, 2); end
score = double(sc(:, pos));
yp = categorical(string(yp), [cfg.NegativeClass cfg.PositiveClass]);
end

function [X2, scaler] = robust_scale_fit(X)
scaler.center = median(X, 1, 'omitnan');
scaler.scale = prctile(X, 75, 1) - prctile(X, 25, 1);
scaler.scale(scaler.scale == 0 | isnan(scaler.scale)) = 1;
X2 = robust_scale_apply(X, scaler);
end

function X2 = robust_scale_apply(X, scaler)
X2 = (X - scaler.center) ./ scaler.scale;
end

function m = classification_metrics(y, yp, score, cfg)
y = categorical(string(y), [cfg.NegativeClass cfg.PositiveClass]);
yp = categorical(string(yp), [cfg.NegativeClass cfg.PositiveClass]);
pos = categorical(cfg.PositiveClass, categories(y));
neg = categorical(cfg.NegativeClass, categories(y));
tp = sum(y == pos & yp == pos); tn = sum(y == neg & yp == neg);
fp = sum(y == neg & yp == pos); fn = sum(y == pos & yp == neg);
m.accuracy = div0(tp+tn, numel(y));
m.sensitivity = div0(tp, tp+fn);
m.specificity = div0(tn, tn+fp);
m.precision_positive = div0(tp, tp+fp);
m.f1_positive = div0(2*tp, 2*tp+fp+fn);
m.balanced_accuracy = mean([m.sensitivity m.specificity], 'omitnan');
try, [~,~,~,m.roc_auc] = perfcurve(y, score, pos); catch, m.roc_auc = NaN; end
m.tp = tp; m.tn = tn; m.fp = fp; m.fn = fn;
end

function m = aggregate_metrics(T, yp, score, cfg, level)
groups = make_aggregate_ids(T, level);
ugroups = unique(groups, 'stable');
y = categorical(strings(numel(ugroups),1), [cfg.NegativeClass cfg.PositiveClass]);
pred = y; sc = zeros(numel(ugroups),1);
for i = 1:numel(ugroups)
    idx = groups == ugroups(i);
    y(i) = mode(T.LABEL(idx));
    pred(i) = mode(yp(idx));
    sc(i) = mean(score(idx), 'omitnan');
end
m = classification_metrics(y, pred, sc, cfg);
end

function ids = make_aggregate_ids(T, level)
switch string(level)
    case "object", ids = T.observation_key;
    otherwise, ids = T.subject_key;
end
ids = string(ids);
end

function score = selection_accuracy(T, yp, sc, cfg)
objectm = aggregate_metrics(T, yp, sc, cfg, "object");
sourcem = aggregate_metrics(T, yp, sc, cfg, "source");
gm = aggregate_metrics(T, yp, sc, cfg, "group");
score = mean([sourcem.accuracy objectm.accuracy gm.accuracy], 'omitnan');
end

function m = group_metrics(T, yp, score, cfg)
groups = unique(T.group_id, 'stable');
y = categorical(strings(numel(groups),1), [cfg.NegativeClass cfg.PositiveClass]);
pred = y; sc = zeros(numel(groups),1);
for i = 1:numel(groups)
    idx = T.group_id == groups(i);
    y(i) = mode(T.LABEL(idx));
    pred(i) = mode(yp(idx));
    sc(i) = mean(score(idx), 'omitnan');
end
m = classification_metrics(y, pred, sc, cfg);
end

function foldId = make_grouped_folds(groups, labels, k, seed)
rng(seed); groups = string(groups); labels = categorical(labels);
ug = unique(groups, 'stable');
k = min(k, min_class_group_count(groups, labels));
if k < 2, error('Insufficient groups per class for grouped validation.'); end
gl = categorical(strings(numel(ug),1), categories(labels)); gs = zeros(numel(ug),1);
for i = 1:numel(ug), idx = groups == ug(i); assert(numel(unique(labels(idx))) == 1, 'Conflicting source labels.'); gl(i) = mode(labels(idx)); gs(i) = sum(idx); end
fg = zeros(numel(ug),1);
for c = 1:numel(categories(labels))
    cls = categories(labels); ii = find(gl == cls{c}); ii = ii(randperm(numel(ii)));
    loads = zeros(k,1);
    for j = 1:numel(ii), [~,f] = min(loads); fg(ii(j)) = f; loads(f)=loads(f)+gs(ii(j)); end
end
foldId = zeros(numel(groups),1);
for i = 1:numel(ug), foldId(groups == ug(i)) = fg(i); end
end

function n = min_class_group_count(groups, labels)
labels = categorical(labels); cats = categories(labels); n = Inf;
for c = 1:numel(cats), n = min(n, numel(unique(string(groups(labels == cats{c}))))); end
if isinf(n), n = 0; end
end

function summary = summarise_dataset(T)
keys = unique(T(:, {'analysis_set','domain','time','LABEL'}), 'rows', 'stable');
summary = table();
for i = 1:height(keys)
    idx = T.analysis_set == keys.analysis_set(i) & T.LABEL == keys.LABEL(i);
    row = table(keys.analysis_set(i), keys.domain(i), keys.time(i), string(keys.LABEL(i)), sum(idx), numel(unique(T.group_id(idx))), ...
        'VariableNames', {'analysis_set','domain','time','label','rows','groups'});
    summary = [summary; row]; %#ok<AGROW>
end
end

function plot_spectral(T, cfg, outDir, name)
fig = figure('Visible','off','Color','w'); hold on;
for lab = [cfg.NegativeClass cfg.PositiveClass]
    idx = T.LABEL == lab;
    plot(cfg.Channels, mean(T{idx,cfg.BandNames},1,'omitnan'), '-o', 'LineWidth', 1.3);
end
legend({char(cfg.NegativeClass), char(cfg.PositiveClass)}, 'Location','best');
title(sprintf('%s spectral curves', name), 'Interpreter','none'); xlabel('Band / channel'); ylabel('Mean value'); grid on;
saveas(fig, fullfile(outDir, 'spectral_curves.png')); close(fig);
end

function plot_importance(imp, cfg, outDir, name)
[~,loc] = ismember(cfg.BandNames, string(imp.band)); imp2 = imp(loc,:);
fig = figure('Visible','off','Color','w'); bar(imp2.permutation_delta_selection_accuracy_mean); grid on;
xticks(1:numel(cfg.BandNames)); xticklabels(string(imp2.channel)); xtickangle(45);
title(sprintf('%s band importance', name), 'Interpreter','none'); ylabel('Source/object/group accuracy decrease');
saveas(fig, fullfile(outDir, 'band_importance.png')); close(fig);
end

function plot_topk(topK, outDir, name)
fig = figure('Visible','off','Color','w'); errorbar(topK.k, topK.selection_accuracy_mean, topK.selection_accuracy_std, '-o'); grid on; ylim([0 1]);
title(sprintf('%s top-k performance', name), 'Interpreter','none'); xlabel('Top-k bands'); ylabel('Source/object/group accuracy');
saveas(fig, fullfile(outDir, 'topK_performance.png')); close(fig);
end

function plot_model_ranking(summary, outDir, name)
fig = figure('Visible','off','Color','w');
vals = summary.selection_accuracy_mean;
b = bar(vals); grid on; ylim([0 1]);
if numel(vals) >= 1
    colors = repmat([0.70 0.70 0.70], numel(vals), 1);
    colors(1,:) = [0.10 0.45 0.80];
    if numel(vals) >= 2, colors(2,:) = [0.20 0.65 0.45]; end
    b.FaceColor = 'flat';
    b.CData = colors;
end
xticks(1:height(summary)); xticklabels(cellstr(summary.model)); xtickangle(35);
ylabel('Source/object/group accuracy');
title(sprintf('%s model ranking', name), 'Interpreter','none');
legend({'Rank 1/2 highlighted'}, 'Location','southoutside');
saveas(fig, fullfile(outDir, 'model_ranking_top2.png')); close(fig);
end

function plot_spectral_by_domain(sets, cfg, outDir, datasetName)
if isempty(sets), return; end
n = numel(unique(arrayfun(@(s) string(s.Table.domain(1)), sets), 'stable'));
nCols = min(4, max(1, ceil(sqrt(n))));
nRows = ceil(n / nCols);
fig = figure('Visible','off','Color','w', 'Position', [100 100 360*nCols 260*nRows]);
tiledlayout(nRows, nCols, 'TileSpacing','compact', 'Padding','compact');
domains = unique(arrayfun(@(s) string(s.Table.domain(1)), sets), 'stable');
for i = 1:numel(domains)
    nexttile; hold on;
    expName = domains(i);
    expSets = sets(arrayfun(@(s) string(s.Table.domain(1)) == expName, sets));
    colors = ordered_time_colors(numel(expSets));
    for j = 1:numel(expSets)
        T = expSets(j).Table;
        timeLabel = sprintf('%gtime', T.time(1));
        idxNeg = T.LABEL == cfg.NegativeClass;
        idxPos = T.LABEL == cfg.PositiveClass;
        plot(cfg.Channels, mean(T{idxNeg,cfg.BandNames},1,'omitnan'), '-', 'Color', colors(j,:), 'LineWidth', 1.0, 'DisplayName', sprintf('%s %s', timeLabel, cfg.NegativeClass));
        plot(cfg.Channels, mean(T{idxPos,cfg.BandNames},1,'omitnan'), '--', 'Color', colors(j,:), 'LineWidth', 1.0, 'DisplayName', sprintf('%s %s', timeLabel, cfg.PositiveClass));
    end
    title(expName, 'Interpreter','none', 'FontSize', 9);
    grid on;
    if i > (nRows-1)*nCols, xlabel('Band / channel'); end
    if mod(i-1, nCols) == 0, ylabel('Mean value'); end
    legend('Location','best', 'FontSize', 6);
end
sgtitle(sprintf('%s spectral curves by domain and day', datasetName), 'Interpreter','none', 'Color','k');
saveas(fig, fullfile(outDir, sprintf('%s_spectral_by_domain_overlay.png', datasetName)));
close(fig);
save_domain_spectral_overlays(sets, cfg, outDir);
end

function save_domain_spectral_overlays(sets, cfg, outDir)
detailDir = fullfile(outDir, 'spectral_by_domain_overlay');
if ~exist(detailDir, 'dir'), mkdir(detailDir); end
domains = unique(arrayfun(@(s) string(s.Table.domain(1)), sets), 'stable');
for i = 1:numel(domains)
    expName = domains(i);
    expSets = sets(arrayfun(@(s) string(s.Table.domain(1)) == expName, sets));
    fig = figure('Visible','off','Color','w', 'Position', [100 100 900 560]);
    hold on;
    colors = ordered_time_colors(numel(expSets));
    for j = 1:numel(expSets)
        T = expSets(j).Table;
        timeLabel = sprintf('%gtime', T.time(1));
        idxNeg = T.LABEL == cfg.NegativeClass;
        idxPos = T.LABEL == cfg.PositiveClass;
        plot(cfg.Channels, mean(T{idxNeg,cfg.BandNames},1,'omitnan'), '-', 'Color', colors(j,:), 'LineWidth', 1.8, 'DisplayName', sprintf('%s %s', timeLabel, cfg.NegativeClass));
        plot(cfg.Channels, mean(T{idxPos,cfg.BandNames},1,'omitnan'), '--', 'Color', colors(j,:), 'LineWidth', 1.8, 'DisplayName', sprintf('%s %s', timeLabel, cfg.PositiveClass));
    end
    title(sprintf('%s spectral curves across days', expName), 'Interpreter','none');
    xlabel('Band / channel'); ylabel('Mean value'); grid on;
    legend('Location','bestoutside');
    saveas(fig, fullfile(detailDir, sprintf('%s_spectral_overlay.png', sanitize_filename(expName))));
    close(fig);
end
end

function plot_cross_heatmap(rows, outDir, datasetName)
modes = unique(rows.cross_mode, 'stable');
for m = 1:numel(modes)
    R = rows(rows.cross_mode == modes(m), :);
    train = unique(R.train_set, 'stable'); test = unique(R.test_set, 'stable');
    M = NaN(numel(test), numel(train));
    for i = 1:height(R)
        ti = find(test == R.test_set(i)); tr = find(train == R.train_set(i));
        M(ti,tr) = R.selection_accuracy(i);
    end
    fig = figure('Visible','off','Color','w'); imagesc(M); colorbar;
    xticks(1:numel(train)); xticklabels(cellstr(train)); xtickangle(45);
    yticks(1:numel(test)); yticklabels(cellstr(test));
    title(sprintf('%s %s source/object/group accuracy', datasetName, modes(m)), 'Interpreter','none');
    xlabel('Train set'); ylabel('Test set');
    saveas(fig, fullfile(outDir, sprintf('%s_%s_heatmap.png', datasetName, modes(m))));
    close(fig);
end
end

function s = params_to_string(params)
fn = fieldnames(params); parts = strings(numel(fn),1);
for i = 1:numel(fn), parts(i) = string(fn{i}) + "=" + string(mat2str(params.(fn{i}))); end
s = strjoin(parts, ';');
end

function params = parse_params(s)
params = struct(); parts = split(string(s), ';');
for i = 1:numel(parts)
    kv = split(parts(i), '='); if numel(kv) ~= 2, continue; end
    params.(char(kv(1))) = str2double(kv(2));
end
end

function s = mode_string(v)
u = unique(string(v), 'stable'); c = zeros(numel(u),1);
for i = 1:numel(u), c(i) = sum(string(v)==u(i)); end
[~,idx] = max(c); s = u(idx);
end

function r = tiedrank_desc(x)
[~,o] = sort(double(x), 'descend'); r = zeros(size(x)); r(o) = 1:numel(x);
end

function z = div0(a,b)
if b == 0, z = NaN; else, z = a/b; end
end

function rank = model_complexity_rank(m)
switch string(m)
    case "logistic", rank = 1;
    case "linear_svm", rank = 2;
    case "random_forest", rank = 3;
    case "boosted_trees", rank = 4;
    case "rbf_svm", rank = 5;
    otherwise, rank = 99;
end
end

function colors = ordered_time_colors(n)
stops = [
    0.17 0.48 0.73
    0.30 0.68 0.82
    0.54 0.76 0.45
    0.93 0.72 0.28
    0.86 0.42 0.25
    0.63 0.24 0.46
    ];
if n <= 1
    colors = stops(1, :);
else
    x = linspace(0, 1, size(stops, 1));
    xi = linspace(0, 1, n);
    colors = interp1(x, stops, xi, 'linear');
end
end

function name = sanitize_filename(name)
name = regexprep(char(name), '[^\w\-]+', '_');
end
