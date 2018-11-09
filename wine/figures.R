red <- read_delim("winequality-red.csv", delim = ";")
red$quality <- factor(red$quality)

red <- read_delim("winequality-white.csv", delim = ";")
red2 <- gather(red, key = variable, value = value, -quality)
red <- red %>% drop_na() 

colnames(red) <- red %>% colnames() %>% str_replace_all(" ","_") 

red <- as.data.frame(red)

red %>% skim() %>% kable()

red %>% cor() %>% corrplot.mixed(upper = "ellipse", tl.cex=.8, tl.pos = 'lt', number.cex = .8)
ggsave("figures/correlation_white.pdf", height = 4, width = 6)

cor(red)
summary(red)
ggplot(red2, aes(x = as.factor(quality), y = value)) + 
  geom_bar(stat="identity",position="dodge") + 
  facet_wrap(~as.factor(variable), scales = "free")
  
ggplot(red2, aes(x = as.factor(quality), y = value)) + 
  theme_tufte(11) +
    geom_bar(stat = "identity", position = "dodge") +
    # geom_smooth() +
    stat_summary(aes(factor(quality), value), fun.y = "mean", geom = "line", group = 1, color = "red") + 
    #stat_smooth(method = "lm") +
    annotate("segment", x=-Inf, xend=Inf, y=-Inf, yend=-Inf, color = "grey") +
  annotate("segment", x=-Inf, xend=-Inf, y=-Inf, yend=Inf, color = "grey") +
    facet_wrap(~variable, scale = "free") +
    xlab("Quality") +
    ylab("Min-Max")
ggsave("figures/barplots_red.pdf", width = 7, height = 4)

red %>% mutate(quality = as.factor(quality)) %>% 
  #select(-c(residual_sugar, free_sulfur_dioxide, total_sulfur_dioxide, chlorides)) %>% 
  ggpairs(aes(color = quality, alpha=0.4),
          columns=1:7,
          lower=list(continuous="points"),
          upper=list(continuous="blank"),
          axisLabels="none", switch="both")

red %>% mutate(quality = as.factor(quality)) %>% 
  ggpairs(aes(color = quality, alpha=0.4),
          # columns=1:7,
          lower=list(continuous="points"),
          upper=list(continuous="blank"),
          axisLabels="none", switch="both")
ggsave("figures/large_corr_plot.pdf", width = 12, height = 8)
