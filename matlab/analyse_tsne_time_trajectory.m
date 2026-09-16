function results = analyse_tsne_time_trajectory(options)
% Adapted from the original analysis implementation; identifiers and I/O generalized.
%ANALYSE_TSNE_TIME_TRAJECTORY Study class difference t-SNE movement across TIME.
%
% This analysis fits one common t-SNE embedding per domain using all
% available TIME values for that domain. Coordinates are therefore
% comparable across TIME within the same domain, unlike independent
% domain-TIME t-SNE plots.

if nargin < 1 || isempty(options)
    options = struct();
end

cfg = config();
rng(cfg.RANDOM_SEED);

opts = defaultOptions(options);
dirs = makeOutputDirs(cfg);

obsTable = load_cached_data("observation");
channels = cfg.expected_channels(:).';
featureNames = "mean_band_" + string(channels);
validateFeatureColumns(obsTable, featureNames);

[Xall, validRows] = getValidFeatureMatrix(obsTable, featureNames);
analysisTable = obsTable(validRows, :);

domains = sort(unique(analysisTable.domain)).';
coordTables = cell(0, 1);
centroidTables = cell(0, 1);
distanceTables = cell(0, 1);
summaryRows = cell(0, 1);

for e = 1:numel(domains)
    expName = domains(e);
    expRows = analysisTable.domain == expName;
    expTable = analysisTable(expRows, :);
    expX = Xall(expRows, :);

    status = "valid";
    skipReason = "";
    timeValues = sort(unique(expTable.time)).';
    class1TimeCount = countTimeByLabel(expTable, "class1");
    class0TimeCount = countTimeByLabel(expTable, "class0");
    perplexity = choosePerplexity(height(expTable), opts, cfg);

    if height(expTable) < opts.MinObservations
        status = "skipped";
        skipReason = "fewer observations than MinObservations";
    elseif numel(timeValues) < opts.MinTimeValues
        status = "skipped";
        skipReason = "fewer TIME values than MinTimeValues";
    elseif class1TimeCount < opts.MinTimeValuesForClass1
        status = "skipped";
        skipReason = "class1 class has too few TIME values";
    elseif perplexity < 2
        status = "skipped";
        skipReason = "automatically selected perplexity is below 2";
    end

    if status == "valid"
        [Z, ~, ~] = standardizeForVisualisation(expX);
        Y = runTsneWithPcaInit(Z, perplexity, cfg);
        coordTable = makeCoordinateTable(expTable, Y, perplexity);
        centroidTable = makeCentroidTable(coordTable);
        distanceTable = makeClassDistanceTable(centroidTable);

        coordTables{end + 1, 1} = coordTable; %#ok<AGROW>
        centroidTables{end + 1, 1} = centroidTable; %#ok<AGROW>
        if ~isempty(distanceTable)
            distanceTables{end + 1, 1} = distanceTable; %#ok<AGROW>
        end

        baseName = "tsne_time_trajectory_" + sanitizeName(expName);
        plotDomainTrajectory(dirs.figures, baseName + "_class1.png", ...
            coordTable, centroidTable, expName, "class1");
        plotCentroidComparison(dirs.figures, baseName + "_centroids.png", ...
            coordTable, centroidTable, expName);
        plotClassDistance(dirs.figures, baseName + "_class_distance.png", ...
            distanceTable, expName);
    end

    summaryRows{end + 1, 1} = table(expName, height(expTable), numel(timeValues), ...
        countSubjectsByLabel(expTable, "class1"), countSubjectsByLabel(expTable, "class0"), ...
        class1TimeCount, class0TimeCount, perplexity, status, skipReason, ...
        'VariableNames', {'domain', 'observations', 'time_values', ...
        'class1_subjects', 'class0_subjects', 'class1_time_values', ...
        'class0_time_values', 'perplexity', 'status', 'skip_reason'});
end

summaryTable = vertcat(summaryRows{:});
coordAll = vertcatOrEmpty(coordTables);
centroidAll = vertcatOrEmpty(centroidTables);
distanceAll = vertcatOrEmpty(distanceTables);

summaryFile = fullfile(dirs.tables, 'tsne_time_trajectory_summary.csv');
coordFile = fullfile(dirs.tables, 'tsne_time_trajectory_coordinates.csv');
centroidFile = fullfile(dirs.tables, 'tsne_time_trajectory_centroids.csv');
distanceFile = fullfile(dirs.tables, 'tsne_time_trajectory_class_distances.csv');

writetable(summaryTable, summaryFile);
if ~isempty(coordAll)
    writetable(coordAll, coordFile);
end
if ~isempty(centroidAll)
    writetable(centroidAll, centroidFile);
end
if ~isempty(distanceAll)
    writetable(distanceAll, distanceFile);
end

results = struct();
results.summary_file = summaryFile;
results.coordinates_file = coordFile;
results.centroids_file = centroidFile;
results.class_distances_file = distanceFile;
results.figure_dir = dirs.figures;
results.valid_domains = sum(summaryTable.status == "valid");
results.skipped_domains = sum(summaryTable.status == "skipped");

fprintf('t-SNE TIME trajectory analysis complete.\n');
fprintf('Valid domains: %d\n', results.valid_domains);
fprintf('Skipped domains: %d\n', results.skipped_domains);
fprintf('Tables saved to: %s\n', dirs.tables);
fprintf('Figures saved to: %s\n', dirs.figures);

end

function opts = defaultOptions(options)
opts = struct();
opts.MinObservations = 8;
opts.MinTimeValues = 2;
opts.MinTimeValuesForClass1 = 2;
opts.Perplexity = [];

fields = fieldnames(options);
for i = 1:numel(fields)
    opts.(fields{i}) = options.(fields{i});
end
end

function dirs = makeOutputDirs(cfg)
dirs = struct();
dirs.figures = fullfile(cfg.paths.figures, 'tsne_time_trajectory');
dirs.tables = fullfile(cfg.paths.tables, 'tsne_time_trajectory');
if ~exist(dirs.figures, 'dir')
    mkdir(dirs.figures);
end
if ~exist(dirs.tables, 'dir')
    mkdir(dirs.tables);
end
end

function validateFeatureColumns(obsTable, featureNames)
missing = setdiff(featureNames, string(obsTable.Properties.VariableNames), 'stable');
if ~isempty(missing)
    error('Observation table is missing required feature columns: %s', strjoin(missing, ', '));
end
end

function [X, validRows] = getValidFeatureMatrix(obsTable, featureNames)
Xraw = obsTable{:, cellstr(featureNames)};
validRows = all(isfinite(Xraw), 2);
X = Xraw(validRows, :);
end

function [Z, mu, sigma] = standardizeForVisualisation(X)
mu = mean(X, 1, 'omitnan');
sigma = std(X, 0, 1, 'omitnan');
sigma(sigma == 0 | ~isfinite(sigma)) = 1;
Z = (X - mu) ./ sigma;
end

function perplexity = choosePerplexity(nSamples, opts, cfg)
if ~isempty(opts.Perplexity)
    requested = opts.Perplexity;
elseif isfield(cfg, 'tsne') && isfield(cfg.tsne, 'Perplexity')
    requested = cfg.tsne.Perplexity;
else
    requested = 20;
end
upperBound = max(1, floor((nSamples - 1) / 3));
perplexity = min(requested, upperBound);
end

function Y = runTsneWithPcaInit(Z, perplexity, cfg)
rng(cfg.RANDOM_SEED);
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

function coordTable = makeCoordinateTable(metaTable, Y, perplexity)
wanted = ["source_file", "dataset_group", "original_class", "analysis_label", ...
    "domain", "time", "object_number", "object_name", "subject_key", ...
    "observation_key", "num_patches", "patch_size"];
coordTable = metaTable(:, cellstr(wanted));
coordTable.tsne_1 = Y(:, 1);
coordTable.tsne_2 = Y(:, 2);
coordTable.perplexity = repmat(perplexity, height(coordTable), 1);
coordTable.embedding_scope = repmat("domain_common_all_time", height(coordTable), 1);
end

function centroidTable = makeCentroidTable(coordTable)
labels = sort(unique(coordTable.analysis_label)).';
rows = cell(0, 1);
for l = 1:numel(labels)
    label = labels(l);
    timeValues = sort(unique(coordTable.time(coordTable.analysis_label == label))).';
    previousCentroid = [];
    for d = 1:numel(timeValues)
        timeValue = timeValues(d);
        idx = coordTable.analysis_label == label & coordTable.time == timeValue;
        xy = [coordTable.tsne_1(idx), coordTable.tsne_2(idx)];
        centroid = mean(xy, 1, 'omitnan');
        spread = mean(sqrt(sum((xy - centroid) .^ 2, 2)), 'omitnan');
        if isempty(previousCentroid)
            distanceFromPrevious = NaN;
        else
            distanceFromPrevious = hypot(centroid(1) - previousCentroid(1), ...
                centroid(2) - previousCentroid(2));
        end
        previousCentroid = centroid;
        rows{end + 1, 1} = table(coordTable.domain(1), timeValue, label, sum(idx), ...
            numel(unique(coordTable.subject_key(idx))), centroid(1), centroid(2), spread, ...
            distanceFromPrevious, ...
            'VariableNames', {'domain', 'time', 'analysis_label', 'observations', ...
            'subjects', 'centroid_tsne_1', 'centroid_tsne_2', 'mean_within_label_spread', ...
            'distance_from_previous_time'});
    end
end
centroidTable = vertcat(rows{:});
end

function distanceTable = makeClassDistanceTable(centroidTable)
timeValues = sort(unique(centroidTable.time)).';
rows = cell(0, 1);
for d = 1:numel(timeValues)
    timeValue = timeValues(d);
    class1 = centroidTable(centroidTable.time == timeValue & centroidTable.analysis_label == "class1", :);
    class0 = centroidTable(centroidTable.time == timeValue & centroidTable.analysis_label == "class0", :);
    if isempty(class1) || isempty(class0)
        continue;
    end
    centroidDistance = hypot(class1.centroid_tsne_1 - class0.centroid_tsne_1, ...
        class1.centroid_tsne_2 - class0.centroid_tsne_2);
    meanSpread = mean([class1.mean_within_label_spread, class0.mean_within_label_spread], 'omitnan');
    separationRatio = centroidDistance / meanSpread;
    rows{end + 1, 1} = table(centroidTable.domain(1), timeValue, ...
        class1.observations, class0.observations, centroidDistance, meanSpread, separationRatio, ...
        'VariableNames', {'domain', 'time', 'class1_observations', ...
        'class0_observations', 'centroid_distance', 'mean_within_label_spread', ...
        'separation_ratio'});
end
distanceTable = vertcatOrEmpty(rows);
end

function plotDomainTrajectory(figureDir, filename, coordTable, centroidTable, expName, focusLabel)
fig = figure('Visible', 'off', 'Position', [100, 100, 1000, 760]);
hold on;
otherRows = coordTable.analysis_label ~= focusLabel;
scatter(coordTable.tsne_1(otherRows), coordTable.tsne_2(otherRows), 35, [0.72, 0.72, 0.72], ...
    '^', 'filled', 'MarkerFaceAlpha', 0.30, 'MarkerEdgeColor', 'none', ...
    'DisplayName', 'other class');

focusRows = coordTable.analysis_label == focusLabel;
focusTime = coordTable.time(focusRows);
cmap = turbo(numel(unique(focusTime)));
timeValues = sort(unique(focusTime)).';
for d = 1:numel(timeValues)
    timeValue = timeValues(d);
    rows = focusRows & coordTable.time == timeValue;
    scatter(coordTable.tsne_1(rows), coordTable.tsne_2(rows), 58, cmap(d, :), ...
        'o', 'filled', 'MarkerFaceAlpha', 0.82, 'MarkerEdgeColor', 'k', ...
        'LineWidth', 0.30, 'DisplayName', sprintf('TIME %s', string(timeValue)));
end

centers = centroidTable(centroidTable.analysis_label == focusLabel, :);
centers = sortrows(centers, 'time');
plot(centers.centroid_tsne_1, centers.centroid_tsne_2, '-k', 'LineWidth', 1.4, ...
    'DisplayName', 'class1 centroid path');
scatter(centers.centroid_tsne_1, centers.centroid_tsne_2, 110, 'k', 'x', ...
    'LineWidth', 1.8, 'HandleVisibility', 'off');
for i = 1:height(centers)
    text(centers.centroid_tsne_1(i), centers.centroid_tsne_2(i), ...
        "  " + string(centers.time(i)), 'FontSize', 9, 'Color', [0.05, 0.05, 0.05], ...
        'FontWeight', 'bold');
end

hold off;
grid on;
xlabel('t-SNE 1');
ylabel('t-SNE 2');
title(sprintf('Common domain t-SNE trajectory: %s class1 across TIME', expName));
legend('Location', 'eastoutside');
saveas(fig, fullfile(figureDir, filename));
close(fig);
end

function plotCentroidComparison(figureDir, filename, coordTable, centroidTable, expName)
fig = figure('Visible', 'off', 'Position', [100, 100, 1000, 760]);
hold on;
colors = classColors();
labels = ["class1", "class0"];
markers = ["o", "^"];
for l = 1:numel(labels)
    label = labels(l);
    rows = coordTable.analysis_label == label;
    colorValue = colors.(char(label));
    scatter(coordTable.tsne_1(rows), coordTable.tsne_2(rows), 34, colorValue, ...
        markers(l), 'filled', 'MarkerFaceAlpha', 0.25, 'MarkerEdgeColor', 'none', ...
        'DisplayName', label + " observations");

    centers = centroidTable(centroidTable.analysis_label == label, :);
    centers = sortrows(centers, 'time');
    plot(centers.centroid_tsne_1, centers.centroid_tsne_2, '-', ...
        'Color', colorValue, 'LineWidth', 1.8, 'DisplayName', label + " centroid path");
    scatter(centers.centroid_tsne_1, centers.centroid_tsne_2, 95, colorValue, ...
        markers(l), 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.55, ...
        'HandleVisibility', 'off');
    for i = 1:height(centers)
        text(centers.centroid_tsne_1(i), centers.centroid_tsne_2(i), ...
            "  " + string(centers.time(i)), 'FontSize', 9, 'Color', [0.05, 0.05, 0.05], ...
            'FontWeight', 'bold');
    end
end
hold off;
grid on;
xlabel('t-SNE 1');
ylabel('t-SNE 2');
title(sprintf('Common domain t-SNE centroids across TIME: %s', expName));
legend('Location', 'eastoutside');
saveas(fig, fullfile(figureDir, filename));
close(fig);
end

function plotClassDistance(figureDir, filename, distanceTable, expName)
if isempty(distanceTable)
    return;
end
distanceTable = sortrows(distanceTable, 'time');
fig = figure('Visible', 'off', 'Position', [100, 100, 900, 560]);
yyaxis left;
plot(distanceTable.time, distanceTable.centroid_distance, '-o', 'LineWidth', 1.6, ...
    'MarkerSize', 6);
ylabel('Class1-class0 centroid distance');
yyaxis right;
plot(distanceTable.time, distanceTable.separation_ratio, '-s', 'LineWidth', 1.6, ...
    'MarkerSize', 6);
ylabel('Separation ratio');
grid on;
xlabel('TIME');
title(sprintf('Common t-SNE class separation across TIME: %s', expName));
legend({'centroid distance', 'distance / within-class spread'}, 'Location', 'best');
saveas(fig, fullfile(figureDir, filename));
close(fig);
end

function n = countTimeByLabel(T, label)
rows = T.analysis_label == label;
if any(rows)
    n = numel(unique(T.time(rows)));
else
    n = 0;
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

function colors = classColors()
colors = struct();
colors.class1 = [0.75, 0.18, 0.16];
colors.class0 = [0.10, 0.42, 0.62];
end

function out = vertcatOrEmpty(tables)
if isempty(tables)
    out = table();
else
    out = vertcat(tables{:});
end
end

function safe = sanitizeName(value)
safe = regexprep(char(value), '[^A-Za-z0-9_]+', '_');
end
