library(riskRegression)
library(prodlim)
library(survival)
library(cmprsk)
library(readr)

# Open data
datasets = c('FRAMINGHAM', 'SYNTHETIC_COMPETING', 'PBC')
terms = list(c(2153.75,4589.5,6620.75),
             c(4.0,12.0,31.0),
             c(3.18760267221553,4.95356477932318,7.45329098674844))

for (i in 1:3) {
    if (i == 3){
        # Open saved train file
        data = read_csv(paste0('data/', datasets[i], '.csv'))
        prediction = matrix(0, nrow = nrow(data), ncol = 2 * 3 + 1) # # outcomes * # Times
        rownames(prediction) = as.numeric(rownames(data)) - 1
        colnames(prediction) = c(terms[i][[1]], terms[i][[1]], 'Use')

        # Cross validation
        for (fold in 0:4) {
            data_folder = subset(data, data[paste0("Fold_", fold)] == "Train")
            # Fit model 
            formula = reformulate(setdiff(colnames(data_folder), c("Time", "Event", "Fold_0", "Fold_1", "Fold_2", "Fold_3", "Fold_4")), response = "Hist(Time, Event)") 
            
            for (outcome in 1:2) {
                model = FGR(formula, data = data_folder, cause = outcome)
                # Predict at the time horizons of interest
                test = (data[paste0("Fold_", fold)] == "Test")
                    prediction[test,((outcome-1)*3+1):(outcome*3)] = 1 - predict(model, subset(data, test), terms[i][[1]], cause = outcome)
                }
                prediction[(data[paste0("Fold_", fold)] == "Test"), 7] = fold
            }
        # Save
        write.csv(prediction, paste0('Results/', datasets[i], '_finegray.csv'))
    }
}