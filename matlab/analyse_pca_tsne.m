function results = analyse_pca_tsne(options)
% Adapted from the original analysis implementation; identifiers and I/O generalized.
%ANALYSE_PCA_TSNE Visualise observation-level spectra with PCA and t-SNE.

if nargin < 1 || isempty(options)
    options = struct();
end

cfg = config();
ensureOutputDirs(cfg);
rng(cfg.RANDOM_SEED);

obsTable = load_cached_data("observation");
channels = cfg.expected_channels(:).';
featureNames = "mean_band_" + string(channels);
validateFeatureColumns(obsTable, featureNames);

runSlowAnalyses = getRunSlowAnalyses(cfg);
runDomainTimeTsne = getRunDomainTimeTsne(cfg);
if isfield(options, 'RUN_SLOW_ANALYSES')
    runSlowAnalyses = logical(options.RUN_SLOW_ANALYSES);
end
if isfield(options, 'RUN_DOMAIN_TIME_TSNE')
    runDomainTimeTsne = logical(options.RUN_DOMAIN_TIME_TSNE);
end
connectSubjects = true;

[X, validRows, invalidReason] = getValidFeatureMatrix(obsTable, featureNames);
analysisTable = obsTable(validRows, :);
if any(~validRows)
    warningFile = fullfile(cfg.paths.logs, 'pca_tsne_warnings.txt');
    writeInvalidFeatureWarnings(warningFile, obsTable, validRows, invalidReason);
else
    warningFile = fullfile(cfg.paths.logs, 'pca_tsne_warnings.txt');
    writeTextFile(warningFile, sprintf('PCA/t-SNE warnings\nGenerated: %s\n\nNo invalid feature rows detected.\n', datestr(now, 31)));
end

if height(analysisTable) < 3
    error('Need at least 3 valid observation-level rows for PCA/t-SNE visualisation.');
end

[Z, mu, sigma] = standardizeForVisualisation(X);

[coeff, score, latent, ~, explained] = pca(Z, 'Rows', 'complete');
pcaCoordTable = makePcaCoordinateTable(analysisTable, score, featureNames);
pcaCoeffTable = makePcaCoefficientTable(coeff, featureNames);
explainedTable = table((1:numel(explained)).', explained(:), cumsum(explained(:)), ...
    'VariableNames', {'component', 'explained_variance_percent', 'cumulative_explained_variance_percent'});

perplexity = chooseTsnePerplexity(height(analysisTable), cfg);
tsneY = runCombinedTsne(Z, perplexity, cfg);
tsneCoordTable = makeTsneCoordinateTable(analysisTable, tsneY, perplexity);

saveTablesAndModels(cfg, pcaCoordTable, pcaCoeffTable, explainedTable, tsneCoordTable, ...
    mu, sigma, latent, featureNames, channels, perplexity);

plotPcaViews(cfg, pcaCoordTable, explainedTable, connectSubjects);
plotTsneViews(cfg, tsneCoordTable);

if runSlowAnalyses
    runPerTimeVisualisations(cfg, analysisTable, Z, featureNames);
end

domainTimeSummaryFile = "";
if runDomainTimeTsne
    domainTimeSummaryFile = runDomainTimeTsneAnalysis(cfg, analysisTable, featureNames, warningFile);
elseif isfield(cfg, 'FAST_MODE') && cfg.FAST_MODE
    fprintf('Skipping domain-by-TIME t-SNE because cfg.FAST_MODE is true and cfg.RUN_DOMAIN_TIME_TSNE is false.\n');
end

results = struct();
results.valid_observations = height(analysisTable);
results.invalid_observations = sum(~validRows);
results.pca_coordinates_file = fullfile(cfg.paths.tables, 'pca_coordinates.csv');
results.pca_coefficients_file = fullfile(cfg.paths.tables, 'pca_coefficients.csv');
results.pca_explained_variance_file = fullfile(cfg.paths.tables, 'pca_explained_variance.csv');
results.tsne_coordinates_file = fullfile(cfg.paths.tables, 'tsne_coordinates.csv');
results.warning_file = warningFile;
results.figure_dir = cfg.paths.figures;
results.tsne_perplexity = perplexity;
results.domain_time_tsne_summary_file = domainTimeSummaryFile;

fprintf('PCA/t-SNE analysis complete.\n');
fprintf('Valid observations used: %d\n', results.valid_observations);
fprintf('Invalid observations excluded: %d\n', results.invalid_observations);
fprintf('t-SNE perplexity: %g\n', results.tsne_perplexity);
fprintf('Tables saved to: %s\n', cfg.paths.tables);
fprintf('Figures saved to: %s\n', cfg.paths.figures);
fprintf('Warnings saved to: %s\n', warningFile);

end

function ensureOutputDirs(cfg)
dirs = {cfg.paths.tables, cfg.paths.logs, cfg.paths.figures, cfg.paths.cache};
for k = 1:numel(dirs)
    if ~exist(dirs{k}, 'dir')
        mkdir(dirs{k});
    end
end
end

function validateFeatureColumns(obsTable, featureNames)
missing = setdiff(featureNames, string(obsTable.Properties.VariableNames), 'stable');
if ~isempty(missing)
    error('Observation table is missing required PCA/t-SNE feature columns: %s', strjoin(missing, ', '));
end
end

function runSlow = getRunSlowAnalyses(cfg)
runSlow = false;
if isfield(cfg, 'RUN_SLOW_ANALYSES')
    runSlow = logical(cfg.RUN_SLOW_ANALYSES);
elseif isfield(cfg, 'analysis') && isfield(cfg.analysis, 'run_optional_slow_analyses')
    runSlow = logical(cfg.analysis.run_optional_slow_analyses);
end
end

function runIt = getRunDomainTimeTsne(cfg)
runIt = false;
if isfield(cfg, 'RUN_DOMAIN_TIME_TSNE')
    runIt = logical(cfg.RUN_DOMAIN_TIME_TSNE);
end
end

function [X, validRows, invalidReason] = getValidFeatureMatrix(obsTable, featureNames)
X = obsTable{:, cellstr(featureNames)};
validRows = true(height(obsTable), 1);
invalidReason = strings(height(obsTable), 1);

for i = 1:height(obsTable)
    row = X(i, :);
    if any(isnan(row))
        validRows(i) = false;
        invalidReason(i) = "NaN feature value";
    elseif any(~isfinite(row))
        validRows(i) = false;
        invalidReason(i) = "non-finite feature value";
    end
end

X = X(validRows, :);
end

function [Z, mu, sigma] = standardizeForVisualisation(X)
mu = mean(X, 1, 'omitnan');
sigma = std(X, 0, 1, 'omitnan');
sigma(sigma == 0 | ~isfinite(sigma)) = 1;
Z = (X - mu) ./ sigma;
end

function tableOut = makePcaCoordinateTable(metaTable, score, featureNames)
scoreNames = "PC" + string(1:size(score, 2));
tableOut = metaColumns(metaTable);
tableOut = [tableOut, array2table(score, 'VariableNames', cellstr(scoreNames))];
tableOut.feature_set = repmat(strjoin(featureNames, ','), height(tableOut), 1);
end

function tableOut = makePcaCoefficientTable(coeff, featureNames)
componentNames = "PC" + string(1:size(coeff, 2));
tableOut = array2table(coeff, 'VariableNames', cellstr(componentNames));
tableOut.feature_name = featureNames(:);
tableOut = movevars(tableOut, 'feature_name', 'Before', 1);
end

function tsneTable = makeTsneCoordinateTable(metaTable, tsneY, perplexity)
tsneTable = metaColumns(metaTable);
tsneTable.tsne1 = tsneY(:, 1);
tsneTable.tsne2 = tsneY(:, 2);
tsneTable.perplexity = repmat(perplexity, height(tsneTable), 1);
end

function out = metaColumns(T)
wanted = ["source_file", "dataset_group", "original_class", "analysis_label", ...
    "domain", "time", "object_number", "object_name", "subject_key", ...
    "observation_key", "num_patches", "patch_size"];
out = T(:, cellstr(wanted));
end

function perplexity = chooseTsnePerplexity(n, cfg)
requested = 20;
if isfield(cfg, 'tsne') && isfield(cfg.tsne, 'Perplexity')
    requested = cfg.tsne.Perplexity;
end
upperBound = max(1, floor((n - 1) / 3));
perplexity = min(requested, upperBound);
perplexity = max(1, perplexity);
end

function Y = runCombinedTsne(Z, perplexity, cfg)
rng(cfg.RANDOM_SEED);
numDims = 2;
if isfield(cfg, 'tsne') && isfield(cfg.tsne, 'NumDimensions')
    numDims = cfg.tsne.NumDimensions;
end
initialDims = min(size(Z, 2), numel(cfg.expected_channels));
if isfield(cfg, 'tsne') && isfield(cfg.tsne, 'InitialPCADimensions')
    initialDims = min(initialDims, cfg.tsne.InitialPCADimensions);
end

try
    Y = tsne(Z, 'NumDimensions', numDims, 'Perplexity', perplexity, ...
        'NumPCAComponents', initialDims, 'Standardize', false);
catch
    Y = tsne(Z, 'NumDimensions', numDims, 'Perplexity', perplexity, ...
        'Standardize', false);
end
end

function saveTablesAndModels(cfg, pcaCoordTable, pcaCoeffTable, explainedTable, tsneCoordTable, ...
    mu, sigma, latent, featureNames, channels, perplexity)
pcaCoordFile = fullfile(cfg.paths.tables, 'pca_coordinates.csv');
pcaCoeffFile = fullfile(cfg.paths.tables, 'pca_coefficients.csv');
explainedFile = fullfile(cfg.paths.tables, 'pca_explained_variance.csv');
tsneFile = fullfile(cfg.paths.tables, 'tsne_coordinates.csv');
modelFile = fullfile(cfg.paths.cache, 'pca_tsne_metadata.mat');

writetable(pcaCoordTable, pcaCoordFile);
writetable(pcaCoeffTable, pcaCoeffFile);
writetable(explainedTable, explainedFile);
writetable(tsneCoordTable, tsneFile);

metadata = struct();
metadata.created = datestr(now, 31);
metadata.standardization_mu = mu;
metadata.standardization_sigma = sigma;
metadata.pca_latent = latent;
metadata.feature_names = featureNames;
metadata.channels = channels;
metadata.tsne_perplexity = perplexity;
save(modelFile, 'metadata');
end

function plotPcaViews(cfg, pcaTable, explainedTable, connectSubjects)
plotEmbeddingByLabel(cfg, pcaTable, 'PC1', 'PC2', 'analysis_label', 'pca_by_analysis_label.png', ...
    'PCA by class1/class0 label', explainedTable, connectSubjects);
plotEmbeddingByLabel(cfg, pcaTable, 'PC1', 'PC2', 'time', 'pca_by_time.png', ...
    'PCA by TIME', explainedTable, connectSubjects);
plotEmbeddingByLabel(cfg, pcaTable, 'PC1', 'PC2', 'domain', 'pca_by_domain.png', ...
    'PCA by domain', explainedTable, connectSubjects);
end

function plotTsneViews(cfg, tsneTable)
plotEmbeddingByLabel(cfg, tsneTable, 'tsne1', 'tsne2', 'analysis_label', 'tsne_by_analysis_label.png', ...
    'Combined observation-level t-SNE by label', [], false);
plotEmbeddingByLabel(cfg, tsneTable, 'tsne1', 'tsne2', 'time', 'tsne_by_time.png', ...
    'Combined observation-level t-SNE by TIME', [], false);
plotEmbeddingByLabel(cfg, tsneTable, 'tsne1', 'tsne2', 'domain', 'tsne_by_domain.png', ...
    'Combined observation-level t-SNE by domain', [], false);
end

function plotEmbeddingByLabel(cfg, T, xVar, yVar, colorVar, filename, titleText, explainedTable, connectSubjects)
fig = figure('Visible', 'off', 'Position', [100, 100, 1000, 760]);
hold on;

if connectSubjects
    connectRepeatedSubjects(T, xVar, yVar);
end

groups = unique(T.(colorVar));
cmap = categoricalColors(numel(groups));
legendHandles = gobjects(numel(groups), 1);
for g = 1:numel(groups)
    rows = T.(colorVar) == groups(g);
    if string(colorVar) == "analysis_label"
        legendMarker = markerForClass(string(groups(g)));
    else
        legendMarker = 'o';
    end
    legendHandles(g) = scatter(NaN, NaN, 52, cmap(g, :), legendMarker, 'filled', ...
        'MarkerFaceAlpha', 0.85, 'MarkerEdgeColor', 'k', 'LineWidth', 0.25, ...
        'DisplayName', string(groups(g)));

    if string(colorVar) == "analysis_label"
        marker = markerForClass(string(groups(g)));
        scatter(T.(xVar)(rows), T.(yVar)(rows), 42, cmap(g, :), marker, 'filled', ...
            'MarkerFaceAlpha', 0.78, 'MarkerEdgeColor', 'k', 'LineWidth', 0.25, ...
            'HandleVisibility', 'off');
    else
        classLabels = ["class1", "class0"];
        for c = 1:numel(classLabels)
            classRows = rows & T.analysis_label == classLabels(c);
            if any(classRows)
                scatter(T.(xVar)(classRows), T.(yVar)(classRows), 42, cmap(g, :), ...
                    markerForClass(classLabels(c)), 'filled', ...
                    'MarkerFaceAlpha', 0.78, 'MarkerEdgeColor', 'k', 'LineWidth', 0.25, ...
                    'HandleVisibility', 'off');
            end
        end
    end
end

hold off;
grid on;
if ~isempty(explainedTable) && xVar == "PC1" && yVar == "PC2"
    xlabel(sprintf('PC1 (%.1f%%)', explainedTable.explained_variance_percent(1)));
    ylabel(sprintf('PC2 (%.1f%%)', explainedTable.explained_variance_percent(2)));
else
    xlabel(xVar);
    ylabel(yVar);
end
title(titleText);
legend(legendHandles, cellstr(string(groups)), 'Location', 'eastoutside');
saveas(fig, fullfile(cfg.paths.figures, filename));
close(fig);
end

function marker = markerForClass(label)
if string(label) == "class0"
    marker = '^';
else
    marker = 'o';
end
end

function connectRepeatedSubjects(T, xVar, yVar)
subjects = unique(T.subject_key);
for s = 1:numel(subjects)
    rows = find(T.subject_key == subjects(s));
    if numel(rows) < 2
        continue;
    end
    oneSubject = T(rows, :);
    if numel(unique(oneSubject.analysis_label)) > 1 || numel(unique(oneSubject.domain)) > 1
        continue;
    end
    oneSubject = sortrows(oneSubject, 'time');
    plot(oneSubject.(xVar), oneSubject.(yVar), '-', 'Color', [0.72, 0.72, 0.72], ...
        'LineWidth', 0.45, 'HandleVisibility', 'off');
end
end

function runPerTimeVisualisations(cfg, analysisTable, Z, featureNames)
timeValues = sort(unique(analysisTable.time)).';
for d = 1:numel(timeValues)
    rows = analysisTable.time == timeValues(d);
    if sum(rows) < 3
        continue;
    end
    subTable = analysisTable(rows, :);
    subZ = Z(rows, :);

    [~, subScore, ~, ~, subExplained] = pca(subZ, 'Rows', 'complete');
    subPca = makePcaCoordinateTable(subTable, subScore, featureNames);
    subExplainedTable = table((1:numel(subExplained)).', subExplained(:), cumsum(subExplained(:)), ...
        'VariableNames', {'component', 'explained_variance_percent', 'cumulative_explained_variance_percent'});
    plotEmbeddingByLabel(cfg, subPca, 'PC1', 'PC2', 'analysis_label', ...
        sprintf('pca_time_%s_by_label.png', sanitizeName(string(timeValues(d)))), ...
        sprintf('Per-TIME PCA, TIME %s', string(timeValues(d))), subExplainedTable, false);

    if sum(rows) >= 10
        perplexity = min(10, max(1, floor((sum(rows) - 1) / 3)));
        rng(cfg.RANDOM_SEED);
        Y = tsne(subZ, 'NumDimensions', 2, 'Perplexity', perplexity, 'Standardize', false);
        subTsne = makeTsneCoordinateTable(subTable, Y, perplexity);
        plotEmbeddingByLabel(cfg, subTsne, 'tsne1', 'tsne2', 'analysis_label', ...
            sprintf('tsne_time_%s_by_label.png', sanitizeName(string(timeValues(d)))), ...
            sprintf('Per-TIME t-SNE, TIME %s', string(timeValues(d))), [], false);
    end
end
end

function summaryFile = runDomainTimeTsneAnalysis(cfg, analysisTable, featureNames, warningFile)
figureDir = fullfile(cfg.paths.figures, 'tsne_by_domain_time');
tableDir = fullfile(cfg.paths.tables, 'tsne_by_domain_time');
if ~exist(figureDir, 'dir')
    mkdir(figureDir);
end
if ~exist(tableDir, 'dir')
    mkdir(tableDir);
end

domains = sort(unique(analysisTable.domain)).';
summaryRows = cell(0, 1);
rowCursor = 0;

appendWarning(warningFile, sprintf('\nDomain-by-TIME t-SNE analysis\n\n'));

for e = 1:numel(domains)
    expName = domains(e);
    expRows = analysisTable.domain == expName;
    timeValues = sort(unique(analysisTable.time(expRows))).';

    for d = 1:numel(timeValues)
        timeValue = timeValues(d);
        subsetRows = expRows & analysisTable.time == timeValue;
        subsetTable = analysisTable(subsetRows, :);
        tSubsetStart = tic;

        [status, skipReason, perplexity, class1Subjects, class0Subjects, totalSubjects] = ...
            evaluateDomainTimeSubset(subsetTable, featureNames, cfg);

        if status == "valid"
            X = subsetTable{:, cellstr(featureNames)};
            [Zsubset, ~, ~] = standardizeForVisualisation(X);
            rng(cfg.RANDOM_SEED);
            Y = runTsneWithPcaInit(Zsubset, perplexity, cfg);

            coordTable = table( ...
                repmat(expName, height(subsetTable), 1), ...
                repmat(timeValue, height(subsetTable), 1), ...
                subsetTable.subject_key, ...
                subsetTable.observation_key, ...
                subsetTable.analysis_label, ...
                Y(:, 1), ...
                Y(:, 2), ...
                repmat(perplexity, height(subsetTable), 1), ...
                repmat(height(subsetTable), height(subsetTable), 1), ...
                repmat(class1Subjects, height(subsetTable), 1), ...
                repmat(class0Subjects, height(subsetTable), 1), ...
                'VariableNames', {'domain', 'time', 'subject_key', 'observation_key', ...
                'analysis_label', 'tsne_1', 'tsne_2', 'perplexity', 'sample_count', ...
                'class1_subject_count', 'class0_subject_count'});

            baseName = "tsne_domain_" + sanitizeName(expName) + "_time_" + sanitizeName(string(timeValue));
            writetable(coordTable, fullfile(tableDir, baseName + ".csv"));
            plotDomainTimeTsne(figureDir, baseName + ".png", coordTable, expName, timeValue);
        else
            appendWarning(warningFile, sprintf('WARNING: Skipped domain=%s, TIME=%s: %s\n', ...
                expName, string(timeValue), skipReason));
        end

        runtimeSeconds = toc(tSubsetStart);
        rowCursor = rowCursor + 1;
        summaryRows{rowCursor, 1} = table(expName, timeValue, totalSubjects, class1Subjects, ...
            class0Subjects, perplexity, status, skipReason, runtimeSeconds, ...
            'VariableNames', {'domain', 'time', 'total_subjects', 'class1_subjects', ...
            'class0_subjects', 'perplexity', 'status', 'skip_reason', 'runtime_seconds'});

        fprintf('Domain-by-TIME t-SNE: domain=%s, TIME=%s, status=%s, runtime=%.2fs\n', ...
            expName, string(timeValue), status, runtimeSeconds);
    end
end

summaryTable = vertcat(summaryRows{:});
summaryFile = fullfile(tableDir, 'tsne_domain_time_summary.csv');
writetable(summaryTable, summaryFile);
end

function [status, skipReason, perplexity, class1Subjects, class0Subjects, totalSubjects] = ...
    evaluateDomainTimeSubset(subsetTable, featureNames, cfg)
status = "valid";
skipReason = "";
nSamples = height(subsetTable);
labels = unique(subsetTable.analysis_label);
class1Subjects = countSubjectsByLabel(subsetTable, "class1");
class0Subjects = countSubjectsByLabel(subsetTable, "class0");
totalSubjects = numel(unique(subsetTable.subject_key));
perplexity = chooseSubsetPerplexity(nSamples, cfg);

if nSamples < 6
    status = "skipped";
    skipReason = "fewer than 6 total observations";
elseif numel(labels) < 2
    status = "skipped";
    skipReason = "only one class is present";
elseif class1Subjects < 2
    status = "skipped";
    skipReason = "class1 has fewer than 2 subjects";
elseif class0Subjects < 2
    status = "skipped";
    skipReason = "class0 has fewer than 2 subjects";
elseif perplexity < 2
    status = "skipped";
    skipReason = "automatically selected perplexity is below 2";
else
    X = subsetTable{:, cellstr(featureNames)};
    featureStd = std(X, 0, 1, 'omitnan');
    if any(~isfinite(X(:))) || any(isnan(X(:)))
        status = "skipped";
        skipReason = "spectral features contain invalid values";
    elseif sum(featureStd > 1e-10) < 2 || rank(X - mean(X, 1, 'omitnan')) < 2
        status = "skipped";
        skipReason = "spectral features contain insufficient valid variation";
    end
end
end

function n = countSubjectsByLabel(T, label)
rows = T.analysis_label == label;
if any(rows)
    n = numel(unique(T.subject_key(rows)));
else
    n = 0;
end
end

function perplexity = chooseSubsetPerplexity(nSamples, cfg)
requested = 20;
if isfield(cfg, 'TSNE_PERPLEXITY')
    requested = cfg.TSNE_PERPLEXITY;
elseif isfield(cfg, 'tsne') && isfield(cfg.tsne, 'Perplexity')
    requested = cfg.tsne.Perplexity;
end
perplexity = min(requested, floor((nSamples - 1) / 3));
end

function Y = runTsneWithPcaInit(Z, perplexity, cfg)
numDims = 2;
if isfield(cfg, 'tsne') && isfield(cfg.tsne, 'NumDimensions')
    numDims = cfg.tsne.NumDimensions;
end

[~, initY] = pca(Z, 'NumComponents', min(2, size(Z, 2)), 'Rows', 'complete');
if size(initY, 2) < 2
    initY(:, 2) = 0;
end
initY = initY(:, 1:2);

try
    Y = tsne(Z, 'NumDimensions', numDims, 'Perplexity', perplexity, ...
        'Standardize', false, 'InitialY', initY);
catch
    Y = tsne(Z, 'NumDimensions', numDims, 'Perplexity', perplexity, ...
        'Standardize', false);
end
end

function plotDomainTimeTsne(figureDir, filename, coordTable, expName, timeValue)
fig = figure('Visible', 'off', 'Position', [100, 100, 900, 760]);
hold on;
colors = classColors();
labels = ["class1", "class0"];
markers = ["o", "^"];
legendHandles = gobjects(numel(labels), 1);
for i = 1:numel(labels)
    rows = coordTable.analysis_label == labels(i);
    legendHandles(i) = scatter(NaN, NaN, 58, colors.(labels(i)), markers(i), 'filled', ...
        'MarkerFaceAlpha', 0.82, 'MarkerEdgeColor', 'k', 'LineWidth', 0.35, ...
        'DisplayName', string(labels(i)));
    if any(rows)
        scatter(coordTable.tsne_1(rows), coordTable.tsne_2(rows), 58, colors.(labels(i)), ...
            markers(i), 'filled', 'MarkerFaceAlpha', 0.82, 'MarkerEdgeColor', 'k', ...
            'LineWidth', 0.35, 'HandleVisibility', 'off');
    end
end
hold off;
grid on;
xlabel('t-SNE 1');
ylabel('t-SNE 2');
title(sprintf('Independent t-SNE: domain %s, TIME %s', expName, string(timeValue)));
legend(legendHandles, cellstr(labels), 'Location', 'best');
ax = gca;
ax.Position = [0.13, 0.13, 0.78, 0.76];
saveas(fig, fullfile(figureDir, filename));
close(fig);
end

function appendWarning(filename, text)
fid = fopen(filename, 'a');
if fid < 0
    error('Could not open warning log for appending: %s', filename);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s', text);
end

function colors = classColors()
colors = struct();
colors.class1 = [0.75, 0.18, 0.16];
colors.class0 = [0.10, 0.42, 0.62];
end

function cmap = categoricalColors(n)
base = [ ...
    0.0000, 0.4470, 0.7410
    0.8500, 0.3250, 0.0980
    0.9290, 0.6940, 0.1250
    0.4940, 0.1840, 0.5560
    0.4660, 0.6740, 0.1880
    0.3010, 0.7450, 0.9330
    0.6350, 0.0780, 0.1840
    0.0000, 0.6000, 0.5000
    0.9000, 0.4000, 0.7000
    0.2000, 0.2000, 0.2000
    0.6500, 0.5000, 0.1000
    0.1000, 0.7000, 0.2000
    0.6000, 0.3000, 0.0000
    0.4000, 0.4000, 0.9000
    0.8000, 0.2000, 0.5000];
if n > 8
    cmap = hsv(n);
elseif n <= size(base, 1)
    cmap = base(1:n, :);
else
    cmap = turbo(n);
end
end

function safe = sanitizeName(value)
safe = regexprep(char(value), '[^A-Za-z0-9_]+', '_');
end

function writeInvalidFeatureWarnings(filename, obsTable, validRows, invalidReason)
fid = fopen(filename, 'w');
if fid < 0
    error('Could not open warning log for writing: %s', filename);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, 'PCA/t-SNE warnings\n');
fprintf(fid, 'Generated: %s\n\n', datestr(now, 31));
badRows = find(~validRows);
for i = 1:numel(badRows)
    idx = badRows(i);
    fprintf(fid, 'WARNING: Excluding observation_key=%s because %s.\n', ...
        obsTable.observation_key(idx), invalidReason(idx));
end
end

function writeTextFile(filename, text)
fid = fopen(filename, 'w');
if fid < 0
    error('Could not open file for writing: %s', filename);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s', text);
end
