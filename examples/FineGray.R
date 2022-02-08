library(riskRegression)
library(prodlim)
library(survival)
library(cmprsk)
library(readr)

# Open data
datasets = c('FRAMINGHAM', 'SYNTHETIC_COMPETING', 'PBC')
terms = list(c(2153.75,4589.5,6620.75),
             c(4.0,12.0,31.0),
             c(2.2656061767604894,3.96106669587121,6.291725988391188))

for (i in 1:3) {
    # Open saved train file
    data = read_csv(paste0('data/', datasets[i], '.csv'))
    prediction = matrix(0, nrow = nrow(data), ncol = 2 * 3 + 1) # # outcomes * # Times
    rownames(prediction) = as.numeric(rownames(data))
    colnames(prediction) = c(terms[i][[1]], terms[i][[1]], 'Use')
    prediction[,-1] = data$Fold

    # Cross validation
    for (fold in 0:4) {
        data_folder = subset(data, data$Fold != fold)
        # Fit model
        formula = reformulate(setdiff(colnames(data_folder), c("Time", "Event", "Fold")), response = "Hist(Time, Event)") 
        
        for (outcome in 1:2) {
            model = FGR(formula, data = data_folder, cause = outcome)
            # Predict at the time horizons of interest
            prediction[(data$Fold == fold),((outcome-1)*3+1):(outcome*3)] = 1 - predict(model, subset(data, data$Fold == fold), terms[i][[1]], cause = outcome)
        }
    }
    # Save
    write.csv(prediction, paste0('Results/', datasets[i], '_finegray.csv'))
}
