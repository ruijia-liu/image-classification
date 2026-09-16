function [rowFold, subjectFoldSummary, finalNumFolds] = make_grouped_folds(subject_key, class_label, requestedNumFolds, randomSeed)
% Adapted from the original analysis implementation; identifiers and I/O generalized.
%MAKE_GROUPED_FOLDS Reproducible grouped, approximately stratified folds.
%
% Stratification is performed at unique-subject level, never at patch-row
% level, so all rows with the same subject_key remain in the same fold.

if nargin < 4 || isempty(randomSeed)
    randomSeed = 42;
end
if nargin < 3 || isempty(requestedNumFolds)
    requestedNumFolds = 5;
end

rng(randomSeed);

subject_key = string(subject_key(:));
class_label = string(class_label(:));

if numel(subject_key) ~= numel(class_label)
    error('subject_key and class_label must have the same number of rows.');
end
if isempty(subject_key)
    error('subject_key and class_label must not be empty.');
end
if requestedNumFolds < 2
    error('requestedNumFolds must be at least 2.');
end

[subjectIds, ~, subjectIndex] = unique(subject_key, 'stable');
nSubjects = numel(subjectIds);
subjectClass = strings(nSubjects, 1);
subjectRowCount = zeros(nSubjects, 1);

for s = 1:nSubjects
    rows = subjectIndex == s;
    labels = unique(class_label(rows));
    labels = labels(strlength(labels) > 0);
    if numel(labels) ~= 1
        error('Subject %s has inconsistent or missing class labels: %s', ...
            subjectIds(s), strjoin(labels, ', '));
    end
    subjectClass(s) = labels(1);
    subjectRowCount(s) = sum(rows);
end

classes = unique(subjectClass, 'stable');
if numel(classes) < 2
    warning('make_grouped_folds:OneClass', ...
        'Only one class is present at subject level. Classification folds will not contain both classes.');
end

classCounts = countSubjectsByClass(subjectClass, classes);
minClassSubjects = min(classCounts);
maxPossibleFolds = min([requestedNumFolds, nSubjects, minClassSubjects]);

if numel(classes) < 2
    maxPossibleFolds = min(requestedNumFolds, nSubjects);
end

if numel(classes) < 2 || minClassSubjects < 2
    error('At least two independent subjects per class are required.');
end
finalNumFolds = maxPossibleFolds;
if finalNumFolds < requestedNumFolds
    warning('make_grouped_folds:ReducedFolds', ...
        'Reduced folds from %d to %d because available subject/class counts are limited.', ...
        requestedNumFolds, finalNumFolds);
end
if nSubjects < requestedNumFolds
    warning('make_grouped_folds:FewSubjects', ...
        'There are fewer unique subjects (%d) than requested folds (%d).', ...
        nSubjects, requestedNumFolds);
end
if numel(classes) >= 2 && minClassSubjects < requestedNumFolds
    warning('make_grouped_folds:FewSubjectsInClass', ...
        'The smallest class has only %d subjects; perfect stratification across %d requested folds is impossible.', ...
        minClassSubjects, requestedNumFolds);
end

subjectFold = assignSubjectFolds(subjectClass, classes, finalNumFolds, randomSeed);
rowFold = subjectFold(subjectIndex);

subjectFoldSummary = table(subjectIds, subjectClass, subjectRowCount, subjectFold, ...
    'VariableNames', {'subject_key', 'class_label', 'row_count', 'fold'});

verifyGroupedFolds(subject_key, class_label, rowFold, subjectFoldSummary, classes, finalNumFolds);
reportFoldBalance(subjectFoldSummary, classes, finalNumFolds);

end

function counts = countSubjectsByClass(subjectClass, classes)
counts = zeros(numel(classes), 1);
for c = 1:numel(classes)
    counts(c) = sum(subjectClass == classes(c));
end
end

function subjectFold = assignSubjectFolds(subjectClass, classes, finalNumFolds, randomSeed)
subjectFold = zeros(numel(subjectClass), 1);
foldClassCounts = zeros(finalNumFolds, numel(classes));
foldTotalCounts = zeros(finalNumFolds, 1);

classOrder = randpermWithSeed(numel(classes), randomSeed + 1000);
for cOrder = 1:numel(classOrder)
    c = classOrder(cOrder);
    classRows = find(subjectClass == classes(c));
    classRows = classRows(randpermWithSeed(numel(classRows), randomSeed + c));

    for i = 1:numel(classRows)
        [~, foldOrder] = sortrows([foldClassCounts(:, c), foldTotalCounts, (1:finalNumFolds).']);
        chosenFold = foldOrder(1);
        subjectFold(classRows(i)) = chosenFold;
        foldClassCounts(chosenFold, c) = foldClassCounts(chosenFold, c) + 1;
        foldTotalCounts(chosenFold) = foldTotalCounts(chosenFold) + 1;
    end
end
end

function order = randpermWithSeed(n, seed)
oldState = rng;
rng(seed);
order = randperm(n);
rng(oldState);
end

function verifyGroupedFolds(subject_key, class_label, rowFold, subjectFoldSummary, classes, finalNumFolds)
if any(rowFold < 1) || any(rowFold > finalNumFolds) || any(isnan(rowFold))
    error('Invalid row fold assignment detected.');
end

subjectIds = unique(subject_key, 'stable');
for s = 1:numel(subjectIds)
    folds = unique(rowFold(subject_key == subjectIds(s)));
    if numel(folds) ~= 1
        error('Subject leakage detected: subject %s appears in multiple folds: %s', ...
            subjectIds(s), mat2str(folds));
    end
end

if height(subjectFoldSummary) ~= numel(subjectIds)
    error('Subject fold summary does not contain exactly one row per unique subject.');
end

for fold = 1:finalNumFolds
    testRows = rowFold == fold;
    trainRows = rowFold ~= fold;
    testClasses = unique(class_label(testRows));
    trainClasses = unique(class_label(trainRows));

    if isempty(testClasses)
        error('Fold %d has no test rows.', fold);
    end
    if numel(classes) >= 2
        missingTrain = setdiff(classes, trainClasses);
        if ~isempty(missingTrain)
            warning('make_grouped_folds:MissingTrainClass', ...
                'Training rows for fold %d are missing class(es): %s.', ...
                fold, strjoin(missingTrain, ', '));
        end

        missingTest = setdiff(classes, testClasses);
        if ~isempty(missingTest)
            warning('make_grouped_folds:MissingTestClass', ...
                'Test rows for fold %d are missing class(es): %s.', ...
                fold, strjoin(missingTest, ', '));
        end
    end
end
end

function reportFoldBalance(subjectFoldSummary, classes, finalNumFolds)
foldTotals = groupcounts(subjectFoldSummary, 'fold');
if max(foldTotals.GroupCount) - min(foldTotals.GroupCount) > 1
    warning('make_grouped_folds:ImbalancedFoldSizes', ...
        'Subject counts are not perfectly balanced across folds.');
end

for c = 1:numel(classes)
    counts = zeros(finalNumFolds, 1);
    for fold = 1:finalNumFolds
        counts(fold) = sum(subjectFoldSummary.fold == fold & subjectFoldSummary.class_label == classes(c));
    end
    if max(counts) - min(counts) > 1
        warning('make_grouped_folds:ImbalancedClassFolds', ...
            'Class %s is not perfectly balanced across folds. Counts: %s', ...
            classes(c), mat2str(counts.'));
    end
end
end
