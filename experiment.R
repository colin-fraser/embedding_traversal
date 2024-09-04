#' ---
#' title: "Exploring embedding space traversal"
#' author: "Colin Fraser"
#' output: github_document
#' ---
#' 
#' Note `{openaiwrapper}` is written by me and not on CRAN. It's installable with
#' `devtools::install_github("colin-fraser/wrapify/examples/openai/")`

library(openaiwrapper)
library(readr)
library(stringr)
library(purrr)
library(tidyverse)
library(gganimate)

parse_words <- function(x, sep = "\\s+") {
  str_split_1(x, sep)
}

embedding_resp_to_vector <- function(emb) {
  emb$data |> 
    map("embedding") |> 
    map(unlist)
}

get_embedding_vecs <- function(x, dimensions = NULL, model = c("text-embedding-3-small", "text-embedding-3-large"),
                              post_process = embedding_resp_to_vector) {
  model <- match.arg(model)
  get_embedding(x, dimensions = dimensions, model = model) |>
    post_process()
}

get_embedding_sequence <- function(x, start=1, stop=20, dimensions=8, model = "text-embedding-3-large",
                                   .verbose = FALSE) {
  # Gets a sequence of embeddings by iteratively chopping the input. E.g.
  # get_embedding_sequence("hello there") returns a list with the embeddings for
  # "hello" and "hello there".
  out <- vector(mode = "list", length = stop - start + 1)
  words <- parse_words(x)[start:stop]
  chunks <- map_chr(start:stop, \(x) paste(words[start:x], collapse = " "))
  get_embedding_vecs(chunks, model = model, dimensions = dimensions)
}

get_embedding_sequence_for_file <- function(file, start, stop, dimensions=8, model = model,
                                            .verbose = FALSE) {
  text <- read_file(file)
  get_embedding_sequence(text, start, stop, dimensions=dimensions, model=model, .verbose=.verbose)
}

diffs <- function(embedding_sequence, p = 2) {
  # computes the distances between each embedding in the embedding sequence
  map_dbl(seq_len(length(embedding_sequence)-1), \(x) {
    sum((embedding_sequence[[x+1]] - embedding_sequence[[x]])^p)^(1/p)
  })
}

convert_to_df <- function(embedding_sequence) {
  do.call("rbind", embedding_sequence) |>
    as_tibble() |>
    mutate(t = row_number())
}

build_embedding_sequence_df <- function(file, start=1, stop=20, p=2, text = NULL, dimensions=8, 
                                        name = NA, model = "text-embedding-3-small", .verbose=TRUE) {
  if (is.null(file)) {
    if (is.null(text)) {
      stop("text and file can't both be null")
    }
    file <- tempfile()
    write_file(text, file)
  }
  text <- read_file(file)
  all_words <- parse_words(text)
  stop <- min(stop, length(all_words))
  words <- all_words[start:stop]
  embedding_sequence <- get_embedding_sequence_for_file(file, start, stop, dimensions=dimensions,
                                                        model = model, .verbose = TRUE)
  embedding_df <- convert_to_df(embedding_sequence)
  embedding_df$word <- words
  embedding_df$delta <- c(NA, diffs(embedding_sequence, p))
  embedding_df$name <- name
  embedding_df
}

if (file.exists("data/essays_with_embeddings.csv")) {
  essays <- read_csv("data/essays_with_embeddings.csv")
} else {
  real_essay_seq <- build_embedding_sequence_df(file="data/happiness_success_person.txt", 1, Inf, 
                                                name = 'Human', dimensions = NULL, 
                                                model = "text-embedding-3-large")
  chatgpt_essay_seq <- build_embedding_sequence_df(file="data/happiness_success_chatgpt.txt", 1, Inf, 
                                                   name = 'ChatGPT', dimensions = NULL, 
                                                   model = "text-embedding-3-large")
  
  essays <- bind_rows(real_essay_seq, chatgpt_essay_seq) |> 
    arrange(name, t) |> 
    group_by(name) |> 
    mutate(cumulative_distance = cumsum(coalesce(delta, 0))) |> 
    ungroup()
  
  pcs <- essays |>
    select(starts_with("V")) |>
    prcomp(rank=100, scale = TRUE)
  
  essays <- pcs$x[,1:5] |>
    as_tibble() |>
    bind_cols(essays)
  
  write_csv(essays, "data/essays_with_embeddings.csv")
}
  

essays |>
  ggplot(aes(x = t, y = cumulative_distance, color = name, label = word)) +
  geom_line() +
  labs(title = "Cumulative (Euclidean) distance through the embedding space",
       x = "Word", y = "Distance", color = "Essay author") +
  theme_minimal()

essays |> 
  ggplot(aes(x = PC1, y=PC2, color = name)) +
  geom_path() +
  transition_reveal(t) +
  labs(title = 'Traversal of the embedding space', subtitle = "(First two principal components)", color = "Text by") +
  theme_minimal()

viz <- essays |> 
  mutate(color = scales::cscale(coalesce(essays$delta, 0), 
                                scales::pal_gradient_n(c('grey90', 'red')), 
                                trans = scales::transform_log()),
         fw = abs(1000 * delta / max(abs(delta), na.rm=TRUE))) |> 
  transmute(name, t, word, delta = coalesce(delta, 0), color,
            tagged_word = str_glue('<span style="color:{color};font-weight:{fw};">{word}</span>')) |> 
  filter(!is.na(word)) |> 
  group_by(name) |> 
  summarise(html=paste0(tagged_word, collapse=" ")) |> 
  mutate(html = str_c("<h3>", name, " essay</h3>", html)) |> 
  pull(html)

#' Below are the essays with each word colored and weighted by the distance
#' that it traverses in the embedding space.

htmltools::HTML(viz[1])
htmltools::HTML(viz[2])

