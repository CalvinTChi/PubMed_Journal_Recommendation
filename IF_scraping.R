##
##  2015 - 12 -31
##  Damiano Fantini
##  Web Scraping: extract IF data from the web (http://www.citefactor.org/)
##
## http://www.biotechworld.it/bioinf/2016/01/02/scraping-impact-factor-data-from-the-web-using-httr-and-regex-in-r/

library(httr)
library(dplyr)

baseAddr = "http://www.citefactor.org/journal-impact-factor-list-2015"
extenAddr = ".html"
sitePages = paste0("_", c("2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18"))
sitePages = c("", sitePages)

jTab = matrix(NA,nrow=0, ncol=3)
colnames(jTab) = c("Full_journal_name", "ISSN", "Impact_factor")
jTab = as.data.frame(jTab)

nJournals = 0

for (page in sitePages) {

  queryAddr= paste0(baseAddr, page,extenAddr)
  sourceHTML=GET(queryAddr)
  sourceHTML= toString(sourceHTML) 
  
  tabStart = regexpr("<th>Journal Name</th>", sourceHTML, fixed = TRUE)
  tabEnd = regexpr("CONTENT ENDS HERE", sourceHTML, fixed = TRUE)
  tabHTML = substr(sourceHTML, tabStart, tabEnd)
  
  ## now, let's extract each row (TR block) of our table
  tabChuncks = unlist(strsplit(tabHTML, "</tr>", fixed=TRUE))
  tabChuncks = tabChuncks[2:(length(tabChuncks) - 1)]
  
  nJournals = nJournals + length(tabChuncks)
  
  for (chunck in tabChuncks) {
    
    chunck = gsub("<b>", "", chunck, fixed=TRUE)
    chunck = gsub("</b>", "", chunck, fixed=TRUE)
    chunck = gsub("\n", "", chunck, fixed=TRUE)
  
    tmp_entries = unlist(strsplit(chunck, "</td>", fixed = TRUE))
  
    jTitle = gsub("<td>","", tmp_entries[2], fixed = TRUE)
    ISSN = gsub("<td>", "", tmp_entries[3], fixed = TRUE)
    jIF = NA
    idx = 4
    while (is.na(jIF) & idx <= 7) {
      jIF = gsub("<td>","", tmp_entries[idx], fixed=TRUE)
      if (jIF == "&nbsp;" | jIF == "-") {
        jIF = NA
        idx = idx + 1
      }
    }
    
    # Remove journals that publish review articles
    if (grepl("review", tolower(jTitle))) {
      next()
    }
    
    jTab = rbind(jTab, data.frame(Full_journal_name=jTitle, ISSN=ISSN, Impact_factor=jIF))
  }
}

print(paste0("Total number of journals: ", nJournals))
print(paste0("Number of journals extracted: ", dim(jTab)[1]))

output_dir = "./data"
write.csv(jTab, paste0(output_dir, "/journal_impact_factors.csv"), quote = FALSE, row.names = FALSE)




