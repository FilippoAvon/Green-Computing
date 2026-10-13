
# Script 3 – Green version in R

suppressPackageStartupMessages({
  library(ranger)       
  library(readr)        
  library(dplyr)        
})



matthews_corrcoef <- function(y_true, y_pred) {
  # Manual MCC 
  tp <- sum(y_true == 1 & y_pred == 1)
  tn <- sum(y_true == 0 & y_pred == 0)
  fp <- sum(y_true == 0 & y_pred == 1)
  fn <- sum(y_true == 1 & y_pred == 0)
  denom <- sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  if (denom == 0) return(0)
  (tp * tn - fp * fn) / denom
}

select_top_k <- function(X, y, k = 10) {
  # ANOVA 
  k   <- min(k, ncol(X))
  f_scores <- sapply(seq_len(ncol(X)), function(j) {
    tryCatch(
      summary(aov(X[, j] ~ factor(y)))[[1]][["F value"]][1],
      error = function(e) 0
    )
  })
  order(f_scores, decreasing = TRUE)[seq_len(k)]
}

scale_matrix <- function(X_tr, X_te) {
  means <- colMeans(X_tr)
  sds   <- apply(X_tr, 2, sd)
  sds[sds == 0] <- 1            # avoid division by zero
  X_tr_s <- sweep(sweep(X_tr, 2, means), 2, sds, "/")
  X_te_s <- sweep(sweep(X_te, 2, means), 2, sds, "/")
  list(train = X_tr_s, test = X_te_s)
}



get_datasets <- function(data_dir) {
  list(
    neuroblastoma = list(
      path   = file.path(data_dir, "10_7717_peerj_5665_dataYM2018_neuroblastoma.csv"),
      target = "outcome",
      drop   = character(0)
    ),
    pediatric_brain_tumor = list(
      path   = file.path(data_dir, "dataset_Belgrade2021_pediatric_brain_tumor_plos_one_0259095_cleaned.csv"),
      target = "Status OS",
      drop   = "OS"
    ),
    colorectal_cancer = list(
      path   = file.path(data_dir, "dataset_Taipei2018_colorectal_cancer_EHRs_plos_one_0200893_final_cleaned.csv"),
      target = "death",
      drop   = c("bian_ma", "interval", "interval_r", "progress")
    ),
    sepsis = list(
      path   = file.path(data_dir, "journal_pone_0148699_S1_Text_Sepsis_SIRS_EDITED.csv"),
      target = "Mortality",
      drop   = "LOS-ICU"
    ),
    depression_heart_failure = list(
      path   = file.path(data_dir, "journal_pone_0158570_S2File_depression_heart_failure.csv"),
      target = "Death (1=Yes, 0=No)",
      drop   = c("Time from HF to Death (days)",
                 "Time from HF to hospitalization (days)",
                 "Hospitalized (1=Yes, 0=No)")
    )
  )
}

load_dataset <- function(cfg) {
  df     <- read_csv(cfg$path, show_col_types = FALSE)
  names(df) <- trimws(names(df))
  target <- trimws(cfg$target)
  drop   <- trimws(cfg$drop)

  y_raw <- as.integer(df[[target]])
  
  uniq <- sort(unique(y_raw[!is.na(y_raw)]))
  y <- match(y_raw, uniq) - 1L
  X <- df %>% select(-all_of(c(target, drop))) %>% as.matrix()
  mode(X) <- "numeric"

  
  for (j in seq_len(ncol(X))) {
    na_idx <- is.na(X[, j])
    if (any(na_idx)) X[na_idx, j] <- median(X[!na_idx, j])
  }
  
  y[is.na(y)] <- as.integer(names(which.max(table(y))))

  list(X = X, y = y)
}



repeated_holdout_green <- function(X, y, n_iter = 100, test_size = 0.30,
                                   n_trees = 10, k = 10, seed = 42) {
  
  top_k <- select_top_k(X, y, k)
  X_sel <- X[, top_k, drop = FALSE]

  mccs <- numeric(n_iter)
  n    <- nrow(X_sel)

  for (i in seq_len(n_iter)) {
    set.seed(seed + i)
    test_idx  <- sample(seq_len(n), size = floor(n * test_size))
    train_idx <- setdiff(seq_len(n), test_idx)

    X_tr <- X_sel[train_idx, , drop = FALSE]
    X_te <- X_sel[test_idx,  , drop = FALSE]
    y_tr <- y[train_idx]
    y_te <- y[test_idx]

    scaled  <- scale_matrix(X_tr, X_te)

    train_df <- as.data.frame(scaled$train)
    
    colnames(train_df) <- paste0("f", seq_len(ncol(train_df)))
    train_df$y <- factor(y_tr)

    rf  <- ranger(
      y ~ ., data = train_df,
      num.trees        = n_trees,
      max.depth        = 8,
      num.threads      = 1,       # single-thread = green
      seed             = seed + i,
      verbose          = FALSE
    )

    test_df  <- as.data.frame(scaled$test)
    colnames(test_df) <- paste0("f", seq_len(ncol(test_df)))
    preds    <- as.integer(as.character(
                  predict(rf, data = test_df)$predictions))
    mccs[i]  <- matthews_corrcoef(y_te, preds)
  }
  mccs
}



main <- function() {
  args     <- commandArgs(trailingOnly = TRUE)
  data_dir <- if (length(args) >= 1) args[1] else getwd()

  datasets <- get_datasets(data_dir)
  all_results <- list()

  cat("\n=== Script 3 – R (Green) ===\n\n")
  t_global_start <- proc.time()

  for (name in names(datasets)) {
    cat(sprintf("Dataset: %s\n", name))
    dat <- load_dataset(datasets[[name]])
    X   <- dat$X
    y   <- dat$y
    cat(sprintf("  Samples=%d, Features=%d, Class balance=%.2f%%\n",
                length(y), ncol(X), mean(y) * 100))

    t0   <- proc.time()
    mccs <- repeated_holdout_green(X, y)
    dt   <- (proc.time() - t0)[["elapsed"]]

    m <- mean(mccs); s <- sd(mccs)
    cat(sprintf("  MCC  = %.4f ± %.4f\n", m, s))
    cat(sprintf("  Time = %.2f s\n\n", dt))

    all_results[[name]] <- data.frame(
      dataset  = name,
      mean_MCC = m,
      std_MCC  = s,
      time_s   = dt
    )
  }

  total_elapsed <- (proc.time() - t_global_start)[["elapsed"]]

  cat(sprintf("=== TIMING SUMMARY (Script 3 – R Green) ===\n"))
  cat(sprintf("  Total wall-clock time: %.2f s\n", total_elapsed))
  cat("  Note: energy (Wh) = (time_s * CPU_TDP_W) / 3600\n")
  cat("        CO2-eq (g)  = energy_Wh * grid_intensity_gCO2/Wh\n\n")

  results_df <- do.call(rbind, all_results)
  out_path   <- file.path(data_dir, "results_script3_R.csv")
  write.csv(results_df, out_path, row.names = FALSE)
  cat(sprintf("Results saved to %s\n", out_path))
}

main()
