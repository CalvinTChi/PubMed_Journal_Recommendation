library(ggplot2)

differences = function(v) {
  estimate = v[1]
  estimates = v[2:length(v)]
  return(sort(estimates - estimate))
}

base_b = read.csv("journal_baseline_performance.csv", header = TRUE)
embed_b = read.csv("embedding_performance.csv", header = TRUE)
multi_b = read.csv("multitask_performance.csv", header = TRUE)

base_diff = apply(base_b, 2, differences)
embed_diff = apply(embed_b, 2, differences)
multi_diff = apply(multi_b, 2, differences)

base_q = apply(base_diff, 2, quantile, c(0.05, 0.95))
embed_q = apply(embed_diff, 2, quantile, c(0.05, 0.95))
multi_q = apply(multi_diff, 2, quantile, c(0.05, 0.95))

df_accuracy <- data.frame(x = c("baseline", "multitask", "embedding"),
                          F = c(base_b[1, 1], multi_b[1, 1], embed_b[1, 1]),
                          L = c(base_b[1, 1] - base_q[2, 1], multi_b[1, 1] - multi_q[2, 1], embed_b[1, 1] - embed_q[2, 1]),
                          U = c(base_b[1, 1] - base_q[1, 1], multi_b[1, 1] - multi_q[1, 1], embed_b[1, 1] - embed_q[1, 1]))

ggplot(df_accuracy, aes(x = x, y = F)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymax = U, ymin = L)) +
  labs(x = "models", y = "accuracy") +
  theme(axis.text = element_text(size = 14),
        axis.title = element_text(size = 16, face = "bold"))
ggsave("pics/accuracy_barplot.png", dpi = 300, units = "cm",
       height = 20, width = 18)

df_auc <- data.frame(x = c("baseline", "multitask", "embedding"),
                     F = c(base_b[1, 2], multi_b[1, 2], embed_b[1, 2]),
                     L = c(base_b[1, 2] - base_q[2, 2], multi_b[1, 2] - multi_q[2, 2], embed_b[1, 2] - embed_q[2, 2]),
                     U = c(base_b[1, 2] - base_q[1, 2], multi_b[1, 2] - multi_q[1, 2], embed_b[1, 2] - embed_q[1, 2]))

ggplot(df_auc, aes(x = x, y = F)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymax = U, ymin = L)) +
  labs(x = "models", y = "AUC") +
  theme(axis.text = element_text(size = 14),
        axis.title = element_text(size = 16, face = "bold"))
ggsave("pics/auc_barplot.png", dpi = 300, units = "cm",
       height = 20, width = 18)




