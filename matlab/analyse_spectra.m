function results = analyse_spectra()
% Adapted from the original analysis implementation; identifiers and I/O generalized.
%ANALYSE_SPECTRA Compare class1 and class0 observation-level spectra.

cfg = config();
ensureOutputDirs(cfg);

rng(cfg.RANDOM_SEED);
obsTable = load_cached_data("observation");

channels = cfg.expected_channels(:).';
bandNames = "mean_band_" + string(channels);
validateObservationBands(obsTable, bandNames);

analysisLabels = ["class1", "class0"];
timeValues = sort(unique(obsTable.time)).';

warningFile = fullfile(cfg.paths.logs, 'spectral_analysis_warnings.txt');
fid = fopen(warningFile, 'w');
if fid < 0
    error('Could not open warning log for writing: %s', warningFile);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, 'Image spectral analysis warnings\n');
fprintf(fid, 'Generated: %s\n\n', datestr(now, 31));

[summaryTable, differenceTable] = computeSpectralTables(obsTable, channels, bandNames, timeValues, analysisLabels, fid);

summaryFile = fullfile(cfg.paths.tables, 'spectral_summary_by_time_class.csv');
differenceFile = fullfile(cfg.paths.tables, 'spectral_class_difference_by_time_band.csv');
writetable(summaryTable, summaryFile);
writetable(differenceTable, differenceFile);

plotSpectraByTime(cfg, summaryTable, channels, timeValues, analysisLabels);
plotDifferenceLines(cfg, differenceTable, channels, timeValues);
plotDifferenceHeatmap(cfg, differenceTable, channels, timeValues);
plotSelectedLongitudinalBands(cfg, summaryTable, channels, analysisLabels);

if isfield(cfg.analysis, 'run_optional_slow_analyses') && cfg.analysis.run_optional_slow_analyses
    plotPerDomainSpectra(cfg, obsTable, channels, bandNames, analysisLabels);
end

results = struct();
results.summary_table = summaryTable;
results.difference_table = differenceTable;
results.summary_file = summaryFile;
results.difference_file = differenceFile;
results.warning_file = warningFile;
results.figure_dir = cfg.paths.figures;

fprintf('Spectral summary saved to: %s\n', summaryFile);
fprintf('Class difference table saved to: %s\n', differenceFile);
fprintf('Spectral analysis warnings saved to: %s\n', warningFile);
fprintf('Spectral figures saved to: %s\n', cfg.paths.figures);

end

function ensureOutputDirs(cfg)
dirs = {cfg.paths.tables, cfg.paths.logs, cfg.paths.figures};
for k = 1:numel(dirs)
    if ~exist(dirs{k}, 'dir')
        mkdir(dirs{k});
    end
end
end

function validateObservationBands(obsTable, bandNames)
missing = setdiff(bandNames, string(obsTable.Properties.VariableNames), 'stable');
if ~isempty(missing)
    error('Observation table is missing required mean band columns: %s', strjoin(missing, ', '));
end
end

function [summaryTable, differenceTable] = computeSpectralTables(obsTable, channels, bandNames, timeValues, analysisLabels, fid)
summaryRows = cell(numel(timeValues) * numel(analysisLabels), 1);
rowCursor = 0;

diffTime = [];
diffChannel = [];
diffClass1Mean = [];
diffClass0Mean = [];
diffValue = [];
diffClass1N = [];
diffClass0N = [];
diffClass1Domains = strings(0, 1);
diffClass0Domains = strings(0, 1);
diffConfounded = false(0, 1);

for d = 1:numel(timeValues)
    timeValue = timeValues(d);
    timeRows = obsTable.time == timeValue;
    classStats = struct();

    fprintf('\nTIME %s comparison\n', string(timeValue));
    fprintf('------------------\n');
    fprintf(fid, 'TIME %s comparison\n', string(timeValue));

    for c = 1:numel(analysisLabels)
        label = analysisLabels(c);
        rows = timeRows & obsTable.analysis_label == label;
        classData = obsTable(rows, :);
        domains = unique(classData.domain);
        subjectCount = numel(unique(classData.subject_key));

        fprintf('%s subjects: %d | domains: %s\n', label, subjectCount, joinOrNone(domains));
        fprintf(fid, '%s subjects: %d | domains: %s\n', label, subjectCount, joinOrNone(domains));

        stats = spectralStats(classData, bandNames);
        classStats.(label) = stats;
        classStats.(label).domains = domains;
        classStats.(label).subject_count = subjectCount;

        rowCursor = rowCursor + 1;
        summaryRows{rowCursor} = makeSummaryRows(timeValue, label, channels, stats, subjectCount, domains);
    end

    class1Domains = classStats.class1.domains;
    class0Domains = classStats.class0.domains;
    confounded = ~isempty(class1Domains) && ~isempty(class0Domains) && ...
        isempty(intersect(class1Domains, class0Domains));
    if confounded
        logWarning(fid, 'TIME %s comparison is confounded: class1 domains [%s], class0 domains [%s].', ...
            string(timeValue), joinOrNone(class1Domains), joinOrNone(class0Domains));
    end
    fprintf(fid, '\n');

    class1Mean = classStats.class1.mean;
    class0Mean = classStats.class0.mean;
    difference = class1Mean - class0Mean;
    nBands = numel(channels);

    diffTime = [diffTime; repmat(timeValue, nBands, 1)]; %#ok<AGROW>
    diffChannel = [diffChannel; channels(:)]; %#ok<AGROW>
    diffClass1Mean = [diffClass1Mean; class1Mean(:)]; %#ok<AGROW>
    diffClass0Mean = [diffClass0Mean; class0Mean(:)]; %#ok<AGROW>
    diffValue = [diffValue; difference(:)]; %#ok<AGROW>
    diffClass1N = [diffClass1N; repmat(classStats.class1.subject_count, nBands, 1)]; %#ok<AGROW>
    diffClass0N = [diffClass0N; repmat(classStats.class0.subject_count, nBands, 1)]; %#ok<AGROW>
    diffClass1Domains = [diffClass1Domains; repmat(joinOrNone(class1Domains), nBands, 1)]; %#ok<AGROW>
    diffClass0Domains = [diffClass0Domains; repmat(joinOrNone(class0Domains), nBands, 1)]; %#ok<AGROW>
    diffConfounded = [diffConfounded; repmat(confounded, nBands, 1)]; %#ok<AGROW>
end

summaryTable = vertcat(summaryRows{1:rowCursor});
differenceTable = table(diffTime, diffChannel, diffClass1Mean, diffClass0Mean, diffValue, ...
    diffClass1N, diffClass0N, diffClass1Domains, diffClass0Domains, diffConfounded, ...
    'VariableNames', {'time', 'channel', 'class1_mean', 'class0_mean', ...
    'class1_minus_class0', 'class1_subject_count', 'class0_subject_count', ...
    'class1_domains', 'class0_domains', 'domain_confounded'});
end

function stats = spectralStats(classData, bandNames)
nBands = numel(bandNames);
stats = struct();
if isempty(classData)
    stats.mean = NaN(1, nBands);
    stats.std = NaN(1, nBands);
    stats.se = NaN(1, nBands);
    stats.ci95_low = NaN(1, nBands);
    stats.ci95_high = NaN(1, nBands);
    stats.n = 0;
    return;
end

subjectTable = subjectMeanSpectra(classData, bandNames);
X = subjectTable{:, cellstr(bandNames)};
stats.n = height(subjectTable);
stats.mean = mean(X, 1, 'omitnan');
stats.std = std(X, 0, 1, 'omitnan');
stats.se = stats.std ./ sqrt(max(stats.n, 1));
if stats.n >= 2
    tValue = tinv(0.975, stats.n - 1);
    stats.ci95_low = stats.mean - tValue .* stats.se;
    stats.ci95_high = stats.mean + tValue .* stats.se;
else
    stats.ci95_low = NaN(1, nBands);
    stats.ci95_high = NaN(1, nBands);
end
end

function subjectTable = subjectMeanSpectra(classData, bandNames)
[g, keys] = findgroups(classData(:, {'subject_key'}));
subjectTable = keys;
for b = 1:numel(bandNames)
    subjectTable.(bandNames(b)) = splitapply(@(x) mean(x, 'omitnan'), classData.(bandNames(b)), g);
end
end

function rows = makeSummaryRows(timeValue, label, channels, stats, subjectCount, domains)
nBands = numel(channels);
rows = table( ...
    repmat(timeValue, nBands, 1), ...
    repmat(label, nBands, 1), ...
    channels(:), ...
    stats.mean(:), ...
    stats.std(:), ...
    stats.se(:), ...
    stats.ci95_low(:), ...
    stats.ci95_high(:), ...
    repmat(subjectCount, nBands, 1), ...
    repmat(joinOrNone(domains), nBands, 1), ...
    'VariableNames', {'time', 'analysis_label', 'channel', 'mean_value', ...
    'std_across_subjects', 'se_across_subjects', 'ci95_low', 'ci95_high', ...
    'unique_subject_count', 'domains_represented'});
end

function plotSpectraByTime(cfg, summaryTable, channels, timeValues, analysisLabels)
nTime = numel(timeValues);
nCols = min(4, nTime);
nRows = ceil(nTime / nCols);
colors = classColors();

fig = figure('Visible', 'off', 'Position', [100, 100, 1400, 280 * nRows]);
tiledlayout(nRows, nCols, 'TileSpacing', 'compact', 'Padding', 'compact');
for d = 1:nTime
    nexttile;
    hold on;
    legendHandles = gobjects(numel(analysisLabels), 1);
    for c = 1:numel(analysisLabels)
        label = analysisLabels(c);
        rows = summaryTable.time == timeValues(d) & summaryTable.analysis_label == label;
        T = sortrows(summaryTable(rows, :), 'channel');
        legendHandles(c) = plotSpectrumWithUncertainty(T.channel, T.mean_value, ...
            T.ci95_low, T.ci95_high, colors.(label), string(label));
    end
    hold off;
    title("TIME " + string(timeValues(d)));
    xlabel('Channel');
    ylabel('Mean band value');
    legend(legendHandles, cellstr(analysisLabels), 'Location', 'best');
    grid on;
end
saveas(fig, fullfile(cfg.paths.figures, 'spectra_by_time.png'));
close(fig);
end

function lineHandle = plotSpectrumWithUncertainty(x, y, low, high, colorValue, displayName)
if all(isfinite(low)) && all(isfinite(high))
    fill([x; flipud(x)], [low; flipud(high)], colorValue, ...
        'FaceAlpha', 0.16, 'EdgeColor', 'none', 'HandleVisibility', 'off');
end
lineHandle = plot(x, y, '-o', 'Color', colorValue, 'LineWidth', 1.4, ...
    'MarkerSize', 3, 'DisplayName', displayName);
end

function plotDifferenceLines(cfg, differenceTable, channels, timeValues)
fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 700]);
hold on;
cmap = categoricalColors(numel(timeValues));
legendHandles = gobjects(numel(timeValues), 1);
for d = 1:numel(timeValues)
    rows = differenceTable.time == timeValues(d);
    T = sortrows(differenceTable(rows, :), 'channel');
    legendHandles(d) = plot(T.channel, T.class1_minus_class0, '-o', ...
        'Color', cmap(d, :), 'DisplayName', "TIME " + string(timeValues(d)), ...
        'LineWidth', 1.1, 'MarkerSize', 3);
end
yline(0, 'k--', 'HandleVisibility', 'off');
hold off;
xlabel('Channel');
ylabel('Class1 mean - class0 mean');
title('Spectral class differences by TIME');
legend(legendHandles, cellstr("TIME " + string(timeValues)), 'Location', 'eastoutside');
grid on;
xlim([min(channels), max(channels)]);
saveas(fig, fullfile(cfg.paths.figures, 'spectral_difference_lines_by_time.png'));
close(fig);
end

function plotDifferenceHeatmap(cfg, differenceTable, channels, timeValues)
Z = NaN(numel(channels), numel(timeValues));
for d = 1:numel(timeValues)
    for b = 1:numel(channels)
        rows = differenceTable.time == timeValues(d) & differenceTable.channel == channels(b);
        if any(rows)
            Z(b, d) = differenceTable.class1_minus_class0(find(rows, 1));
        end
    end
end

fig = figure('Visible', 'off', 'Position', [100, 100, 950, 650]);
imagesc(timeValues, channels, Z);
set(gca, 'YDir', 'normal');
colorbar;
xlabel('TIME');
ylabel('Channel');
title('Class1 mean - class0 mean heatmap');
saveas(fig, fullfile(cfg.paths.figures, 'spectral_difference_heatmap_band_by_time.png'));
close(fig);
end

function plotSelectedLongitudinalBands(cfg, summaryTable, channels, analysisLabels)
selected = channels(round(linspace(1, numel(channels), min(4, numel(channels)))));
colors = classColors();

fig = figure('Visible', 'off', 'Position', [100, 100, 1100, 750]);
tiledlayout(2, ceil(numel(selected) / 2), 'TileSpacing', 'compact', 'Padding', 'compact');
for b = 1:numel(selected)
    nexttile;
    hold on;
    legendHandles = gobjects(numel(analysisLabels), 1);
    for c = 1:numel(analysisLabels)
        label = analysisLabels(c);
        rows = summaryTable.channel == selected(b) & summaryTable.analysis_label == label;
        T = sortrows(summaryTable(rows, :), 'time');
        if all(isfinite(T.ci95_low)) && all(isfinite(T.ci95_high))
            legendHandles(c) = errorbar(T.time, T.mean_value, T.mean_value - T.ci95_low, T.ci95_high - T.mean_value, ...
                '-o', 'Color', colors.(label), 'LineWidth', 1.2, 'MarkerSize', 4, 'DisplayName', string(label));
        else
            legendHandles(c) = errorbar(T.time, T.mean_value, T.se_across_subjects, ...
                '-o', 'Color', colors.(label), 'LineWidth', 1.2, 'MarkerSize', 4, 'DisplayName', string(label));
        end
    end
    hold off;
    title(string(selected(b)) + " channel");
    xlabel('TIME');
    ylabel('Mean band value');
    legend(legendHandles, cellstr(analysisLabels), 'Location', 'best');
    grid on;
end
saveas(fig, fullfile(cfg.paths.figures, 'spectral_longitudinal_selected_bands.png'));
close(fig);
end

function plotPerDomainSpectra(cfg, obsTable, channels, bandNames, analysisLabels)
domains = unique(obsTable.domain);
colors = classColors();
fig = figure('Visible', 'off', 'Position', [100, 100, 1400, 900]);
tiledlayout(ceil(numel(domains) / 3), 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for e = 1:numel(domains)
    nexttile;
    hold on;
    for c = 1:numel(analysisLabels)
        rows = obsTable.domain == domains(e) & obsTable.analysis_label == analysisLabels(c);
        if any(rows)
            subjectTable = subjectMeanSpectra(obsTable(rows, :), bandNames);
            y = mean(subjectTable{:, cellstr(bandNames)}, 1, 'omitnan');
            plot(channels, y, '-o', 'Color', colors.(analysisLabels(c)), 'LineWidth', 1.2, 'MarkerSize', 3);
        end
    end
    hold off;
    title(domains(e));
    xlabel('Channel');
    ylabel('Mean band value');
    grid on;
end
saveas(fig, fullfile(cfg.paths.figures, 'spectra_by_domain_optional.png'));
close(fig);
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
