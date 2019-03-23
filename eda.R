library(ggplot2)
library(scales)
library(dplyr)
DATA_DIR = "./data"

metadata = read.table(paste0(DATA_DIR, "/metadata.txt"), stringsAsFactors = FALSE, sep = "\t", header = TRUE)

# Topical distribution
ggplot(data = metadata, aes(x = category)) + geom_bar(aes(y = (..count..) / sum(..count..))) +
  scale_x_discrete(name ="Topic", labels = c("bioinformatics", "developmental\n genetics", "epigenetics",
                                             "Mendelian\n phenotypes", "omics\n technologies", "population\n genetics",
                                             "statistical\n genetics", "genome\n structure")) +
  labs(y = "Proportion") + 
  theme(axis.text.x = element_text(size = 20), 
        axis.title.x = element_text(size = 50, margin = margin(t = 20, r = 0, b = 0, l = 0)),
        axis.title.y = element_text(size = 50, margin = margin(t = 0, r = 20, b = 0, l = 0)),
        axis.text.y = element_text(size = 30))

ggsave(filename = "pics/class_distribution.png", width = 16, height = 8,
       dpi = 300)

# Distribution of most popular journals
f = metadata %>% count(journalAbbrev, sort = TRUE) %>% as.data.frame()
f$journalAbbrev = factor(f$journalAbbrev, levels = f$journalAbbrev[order(f$n, decreasing = TRUE)])
f_subset = f[1:5, ]
ggplot(f_subset) + geom_bar(aes(x = journalAbbrev, y = n), stat = "identity") + labs(x = "Journal") +
  theme(axis.title.x = element_text(size = 50),
        axis.title.y = element_text(size = 50),
        axis.text.x = element_text(size = 20),
        axis.text.y = element_text(size = 30)) +
  labs(y = "Count")
ggsave(filename = "pics/journal_popularity.png", width = 17, height = 8,
       dpi = 300)

print(length(unique(metadata$journalAbbrev)))

# Distribution of impact factor
ggplot(metadata, aes(x=impact_factor)) + geom_histogram() + labs(x = "Impact Factor") +
  theme(axis.title.x = element_text(size = 50),
        axis.title.y = element_text(size = 50),
        axis.text.x = element_text(size = 30),
        axis.text.y = element_text(size = 30))

ggsave(filename = "pics/impact_factor_distribution.png", width = 17, height = 8,
       dpi = 300)


