# %%
import similarity_measure
import numpy as np

paragraph1 = """The history of all hitherto existing societies is the history of class struggles.

Freeman and slave, patrician and plebeian, lord and serf, guild master and journeyman, in a word, oppressor and 
oppressed, stood in constant opposition to one another, carried on an uninterrupted, now hidden, now open fight, 
a fight that each time ended, either in a revolutionary reconstitution of society at large, or in the common ruin of 
the contending classes.

In the earlier epochs of history, we find almost everywhere a complicated arrangement of society into various orders, a 
manifold gradation of social rank. In ancient Rome we have patricians, knights, plebeians, slaves; in the Middle Ages, 
feudal lords, vassals, guild masters, journeymen, apprentices, serfs; in almost all of these classes, again, 
subordinate gradations.
 """

paragraph2 = """ Owing to their historical position, it became the vocation of the aristocracies of France and England 
to write pamphlets against modern bourgeois society. In the French revolution of July 1830, and in the English reform 
agitation, these aristocracies again succumbed to the hateful upstart. Thenceforth, a serious political contest was 
altogether out of the question. A literary battle alone remained possible. But even in the domain of literature the old 
cries of the restoration period had become impossible.

In order to arouse sympathy, the aristocracy were obliged to lose sight, apparently, of their own interests, and to 
formulate their indictment against the bourgeoisie in the interest of the exploited working class alone. Thus the 
aristocracy took their revenge by singing lampoons on their new master, and whispering in his ears sinister prophecies 
of coming catastrophe.
"""

paragraph3 = """There was enthusiastic cheering and stamping of feet. Napoleon was so gratified that he left his place 
and came round the table to clink his mug against Mr. Pilkington's before emptying it. When the cheering had died down, 
Napoleon, who had remained on his feet, intimated that he too had a few words to say. 

Like all of Napoleon's speeches, it was short and to the point. He too, he said, was happy that the 
period of misunderstanding was at an end. For a long time there had been rumours circulated, he had reason to think, by 
some malignant enemy that there was something subversive and even revolutionary in the outlook of himself and his 
colleagues. They had been credited with attempting to stir up rebellion among the animals on neighbouring farms. 
Nothing could be further from the truth! Their sole wish, now and in the past, was to live at peace and in normal 
business relations with their neighbours. This farm which he had the honour to control, he added, was a cooperative 
enterprise. The title deeds, which were in his own possession, were owned by the pigs jointly.
"""

paragraph4 = """He had only one criticism, he said, to make of Mr. Pilkington's excellent and neighbourly speech. Mr. 
Pilkington had referred throughout to "Animal Farm." He could not of course know for he, Napoleon, was only now for the 
first time announcing it that the name "Animal Farm" had been abolished. Henceforward the farm was to be known as 
"The Manor Farm" which, he believed, was its correct and original name.

"Gentlemen," concluded Napoleon, "I will give you the same toast as before, but in a different form. Fill your glasses 
to the brim. Gentlemen, here is my toast: To the prosperity of The Manor Farm! "
"""

paragraph5 = """
'Now I will tell you the answer to my question. It is this. The Party seeks power entirely for its own sake. We are not 
interested in the good of others; we are interested solely in power. Not wealth or luxury or long life or happiness: 
only power, pure power. What pure power means you will understand presently. We are different from all the oligarchies 
of the past, in that we know what we are doing. All the others, even those who resembled ourselves, were cowards and 
hypocrites. The German Nazis and the Russian Communists came very close to us in their methods, but they never had the 
courage to recognize their own motives. They pretended, perhaps they even believed, that they had seized power 
unwillingly and for a limited time, and that just round the corner there lay a paradise where human beings would be 
free and equal. We are not like that. We know that no one ever seizes power with the intention of relinquishing it. 
Power is not a means, it is an end. One does not establish a dictatorship in order to safeguard a revolution; one makes 
the revolution in order to establish the dictatorship. The object of persecution is persecution. The object of torture 
is torture. The object of power is power. Now do you begin to understand me?' 
"""

# %%
def get_similarity_matrix(matrix: np.ndarray):
    shape = (matrix.shape[1], matrix.shape[1])

    dot_similarity = np.zeros(shape)
    cosine_similarity = np.zeros(shape)

    contributions = [[[] for _ in range(matrix.shape[1])] for _ in range(matrix.shape[1])]

    for i in range(matrix.shape[1]):
        for j in range(i, matrix.shape[1]):
            data = similarity_measure.compute_similarity(matrix, i, j)

            dot_similarity[i,j] = data["dot_similarity"]
            cosine_similarity[i,j] = data["cosine_similarity"]
            contributions[i][j] = data["contributions"]

    return dot_similarity, cosine_similarity, contributions


def print_pairwise_distance(pairs):
    for i in range(4, -1, -1):
        print(f"Para{i+1}", end=" ")
        for j in range(4, i - 1, -1):
            print(f"{pairs[i, j]:0.3f}", end=" ")
        print()
    print("      ", end="")
    for j in range(4, -1, -1):
        print(f"{'para' + str(j+1) : <5}", end=" ")
    print()


def print_mat(mat):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            print(f"{mat[i,j]:0.3f}", end=" ")
        print()


def main() -> None:
    words = set()
    paragraphs = [paragraph1, paragraph2, paragraph3, paragraph4, paragraph5]
    extracted_paragraphs = []
    words_per_para = None


    for para in paragraphs:
        extract = similarity_measure.extract_alphanumerical(para)[:words_per_para]
        words.update(similarity_measure.get_unique_words(extract))
        extracted_paragraphs.append(extract)

    words = list(words)

    frequency_arr = []

    for para in extracted_paragraphs:
        frequency_arr.append(similarity_measure.get_frequency_vector(para, words))

    matrix = np.hstack(frequency_arr)

    dot, cosine, contributions = get_similarity_matrix(matrix)

    print_pairwise_distance(dot)
    print_pairwise_distance(cosine)

    for i in range(4, -1, -1):
        print(f"Para{i+1}", end=" ")
        for j in range(4, i - 1, -1):
            s = ""
            for idx in contributions[i][j]:
                s = s + " " + words[idx]
            print(f"{s:<18}", end=" ")
        print()

    print("       ", end="")
    for j in range(4, -1, -1):
        print(f"{'para' + str(j+1) : <18}", end=" ")




if __name__ == "__main__":
    main()
