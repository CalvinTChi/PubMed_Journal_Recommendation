library(dplyr)
library(sqldf)

DATA_DIR = "./data"

impact_factor = read.csv(paste0(DATA_DIR, "/journal_impact_factors.csv"), header = TRUE, stringsAsFactors = FALSE)
journal_names = readLines(paste0(DATA_DIR, "/J_Medline.txt"))

# PROCESS JOURNAL NAMES
idx = (grepl("JournalTitle", journal_names) | grepl("ISSN (Print)", journal_names, fixed = TRUE) |grepl("MedAbbr", journal_names) | 
         grepl("ISSN (Online)", journal_names, fixed = TRUE))

journal_names_subset = journal_names[idx]

nJournals = length(journal_names_subset) / 4

# Get full journal names
jIdx = seq(1, length(journal_names_subset) - 3, 4)
jTitle = unlist(strsplit(journal_names_subset[jIdx], "JournalTitle:"))
jTitle = jTitle[seq(2, length(jTitle), 2)]
jTitle = substr(jTitle, 2, nchar(jTitle))

# Get journal abbreviations
abbrev = unlist(strsplit(journal_names_subset[jIdx + 1], "MedAbbr:"))
abbrev = abbrev[seq(2, length(abbrev), 2)]
abbrev = substr(abbrev, 2, nchar(abbrev))

# Get ISSN (Print)
issnPrint = unlist(strsplit(journal_names_subset[jIdx + 2], "ISSN \\(Print\\):"))
issnPrint = issnPrint[seq(2, length(issnPrint), 2)]
issnPrint = substr(issnPrint, 2, nchar(issnPrint))

# GET ISSN (Online)
issnOnline = unlist(strsplit(journal_names_subset[jIdx + 3], "ISSN \\(Online\\):"))
issnOnline = issnOnline[seq(2, length(issnOnline), 2)]
issnOnline = substr(issnOnline, 2, nchar(issnOnline))

jTab = data.frame(Full_journal_name_ncbi = jTitle, abbreviation = abbrev, issnPrint = issnPrint, 
                  issnOnline = issnOnline, stringsAsFactors = FALSE)

idx = which(jTab$issnPrint != "" | jTab$issnOnline != "")
nMissingISSN = length(which(jTab$issnPrint == "" & jTab$issnOnline == ""))
jTab = jTab[idx, ]

# Combine NCBI data source and impact factor data source

tab = sqldf('SELECT A.Full_journal_name_ncbi, A.abbreviation, A.issnPrint, A.issnOnline, B.Impact_factor FROM jTab A INNER JOIN impact_factor B ON A.issnPrint = B.ISSN OR A.issnOnline = B.ISSN')


write.table(tab, paste0(DATA_DIR, "/journals.txt"), quote = FALSE, row.names = FALSE, 
            col.names = TRUE, sep = "\t")

# Example reading usage
#tabTest = read.table(paste0(DATA_DIR, "/journals.txt"), sep = "\t", stringsAsFactors = FALSE, header = TRUE, dec = NULL,
#                     quote = NULL)


