"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass




def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    # this is a underteministic variation of improved score
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    
    weight = random.random()
    
    return float(weight*own_moves - (1-weight)*opp_moves)



def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    # this is a defensive variation of improved score
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    

    return float(pow(own_moves-9, 3) - pow(opp_moves, 3))

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    # this is a look-ahead-score
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    def two_steps_ahead(game, player):
        r, c = game.get_player_location(player)
        directions = [(1, 3), (-2, 0), (-3, -1), (3, -3), (-1, -3), (-4, 2), (-3, 3), (1, -1), (4, -2), (2, -4), (-1, 1), (3, 1), (-3, -3), (2, 0), (0, 4), (1, -3), (4, 0), (1, 1), (0, -4), (-1, -1), (-3, 1), (-4, 0), (4, 2), (3, 3), (-4, -2), (-1, 3), (0, -2), (3, -1), (2, 4), (-2, -4), (-2, 4), (0, 2)]
        potential_two_steps_ahead_moves = [(r + dr, c + dc) for dr, dc in directions if game.move_is_legal((r + dr, c + dc))]
        return potential_two_steps_ahead_moves

    own_potential = len(two_steps_ahead(game, player))
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    
    return float(own_potential + pow(own_moves, 2) - pow(opp_moves, 3))
    

def custom_score_4(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    # this heuristic just focus on player's own performance
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    
    return float(own_moves)


def custom_score_5(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    # this is inspired by Monte Carlo tree search
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    child_node = game.get_legal_moves()
    
    random_no = int(random.random()*len(child_node))
    
    further_game = game.forecast_move(child_node[random_no])
    
    return custom_score_4(further_game, player)

def custom_score_6(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    # this is a greedy and aggressive variation of improved score
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    

    return float(pow(own_moves, 3) - pow(opp_moves, 3))

def custom_score_7(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    # this is a more greedy and aggressive variation of improved score
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    

    return float(pow(own_moves, 5) - pow(opp_moves, 5))


def custom_score_8(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    # this is a less greedy but more defensive variation of improved score
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    

    return float(pow(9-own_moves, 3) - pow(9-opp_moves, 3))

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """
    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).
        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.
        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        legal_moves = game.get_legal_moves()
        if legal_moves:
            best_move = legal_moves[0]
        else:
            best_move = (-1, -1) 

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
        #initialize best_score and best_move
        best_score = float("-inf")
        child_nodes = game.get_legal_moves(game.active_player)
        if child_nodes:
            best_move = child_nodes[0]
        else:
            best_move = (-1, -1)
        #expand each child node
        for move in child_nodes:
            #in this zero-sum game, the higher the score our opponent give to a node,
            #the lower the score we should give to it
            #this is done with the helper function self.evaluate_minimax()
            test_score = - self.evaluate_minimax(game.forecast_move(move), depth - 1, current_player = False)
            #improve our best_score and best_move whenever possible
            if test_score > best_score:
                best_move = move
                best_score = test_score
        return best_move
        
    
    
    def evaluate_minimax(self, game, depth, current_player):
        
        """
        By the identity max(a, b) = −min(−a, −b), we are able to combine the min_value 
        and max_value function into one minimax function. This form of minimax search 
        is sometimes referred as Negamax.
        
        
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        current_player: A boolean indicate where the node visiting belong to the current player. 
                        It osillate as we go deeper into the search tree. When True, we are at
                        a maximizing node. Otherwise, we are at a minimizing node.

        Returns
        -------
        best_score: int
            minimax search evaluated by self.score() 
        """
        
        #timer check
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        #return the score of a node if it reaches the said depth or if it terminate the game
        if depth == 0 or len(game.get_legal_moves()) == 0:
            if current_player:
                return self.score(game, game.active_player)
            return - self.score(game, game.inactive_player)
            
        #let's give a lower bound to the possible score
        best_score = float("-inf")
        #look at the child nodes
        #this is the recurive part with depth decreasing gradually:
        for move in game.get_legal_moves(game.active_player):
            #in this zero-sum game, the higher the score our opponent give to a node,
            #the lower the score we should give to it
            test_score = - self.evaluate_minimax(game.forecast_move(move), depth - 1, not current_player)
            #improve our best_score whenever possible
            if test_score > best_score:
                best_score = test_score
        return best_score
    

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        '''
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return (-1, -1)
        return legal_moves[random.randint(0, len(legal_moves) - 1)]
        '''
        
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        
            
        self.time_left = time_left

        # TODO: finish this function!
        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        '''
        #uncomment this part for a stronger agent that use the winning strategy whenever it is paleyed first
        #each player should try to occupy the center place in their initial move whenever possible
        if game.move_count < 2:
            w = game.width
            h = game.height
            if w%2 == 1 and h%2 == 1:
                if (w//2, h//2) in game.get_blank_spaces():
                    return (w//2, h//2)
        
        #always try to get the the opposite of opponent's direction
        #first, find opponent's location
        opponent_location = game.get_player_location(game._inactive_player)
        #then, rotate that by 180 degree
        try:
            rotated_location = (game.width - opponent_location[0], game.height - opponent_location[1])
            #if that location is availabe and not too bad, go there
            if rotated_location in game.get_legal_moves():
                if game.forecast_move(rotated_location).get_legal_moves():
                    return rotated_location
        except:
            print('opponent_location', opponent_location)
            pass
        '''
        #we are ready to preform alpha-beta pruning here
        
        # Initialize the best move so that this function returns something
        legal_moves = game.get_legal_moves()
        if legal_moves:
            best_move = legal_moves[0]
        else:
            best_move = ( -1, -1)
        depth = 1
        while True:
            try:
                # The try/except block will automatically catch the exception
                # raised when the timer is about to expire.
                best_move = self.alphabeta(game, depth, alpha=float("-inf"), beta=float("inf"))
                depth += 1
            except SearchTimeout:
                return best_move  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move
    
    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        # TODO: finish this function!
        #initialize best_score and best_move
        best_score = float("-inf")
        child_nodes = game.get_legal_moves()
        if child_nodes:
            best_move = child_nodes[0]
        else:
            #print(child_nodes)
            #print(game.get_legal_moves(game.active_player))
            best_move = (-1, -1)
        #sort child_nodes before alpha-beta prunning
        #sorted(child_nodes, key = lambda x : self.score(game.forecast_move(x), game.inactive_player), reverse=True)
        #expand each child node
        for move in child_nodes:
            #evaluate_alphabeta() is a helper function
            #in each level depper, depth decreased by 1 and we switch the perspective from player to his/her opponent
            #note that the role of of alpha and beta have to change accordingly
            #in this zero-sum game, the higher the score our opponent give to a node,
            #the lower the score we should give to it
            next_game = game.forecast_move(move)
            test_score = - self.evaluate_alphabeta(next_game, depth-1, -beta, -alpha, current_player = False)
            #improve our best_score and best_move whenever possible
            if test_score > best_score:
                best_move = move
                best_score = test_score
            #update the record of alpha whenever necessary
            #value of beta will be updated automatically in recursive call of self.evaluate_alphabeta()
            #note that pruning is not preformed in this first layer
            alpha = max(best_score, alpha)
        return best_move
    
    def evaluate_alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), current_player = False):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        #print(game.active_player)
        #print('depth=', depth,'alpha=', alpha,'beta=', beta,'active_score=', self.score(game, game.active_player),'inactive_score=', self.score(game, game.inactive_player))
        #print(game.print_board())
            
        """AlphaBeta pruning is an improved version of MiniMax giving the same output, 
        but in a shorter time (observable in an exponential scale). The trick is to 
        keep records of lower bound of player's best possible score, known as alpha,
        as we search with tree; and an upper bound of player's worst possible score, known
        as beta at the same time. In any branch with alpha no less than beta, it can 
        be pruned. Since we forecast two players with the pre-assumed strategy would not 
        enter such branch, these branch will also be ignored eventually under the usual 
        minimax algorithm, but after a longer time. The branching factor can be cutted 
        down to a half by this method in the average case.
        
        
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
            
        alpha : float
            Alpha is a lower bound of player's best possible score at each node
            At each node, alpha is monotonically increasing (starting from from -inf) as we proceed the search in the branch under it
            
        beta : float
            Beta is an upper bound of player's best possible score at each node
            At each node, beta is monotonically decreasing (starting from from inf) as we proceed the search in the branch under it
        
        current_player: A boolean indicate where the node visiting belong to the current player. 
                        It alternate as we go deeper into the search tree. When True, we are at
                        a maximizing node. Otherwise, we are at a minimizing node.

        Returns
        -------
        best_score: int
            minimax search with alpha-beta pruning evaluated by self.score() 
        """
        #return the score of a node if it reaches the said depth or if it terminate the game
        if depth == 0 or len(game.get_legal_moves()) == 0:
            if current_player:
                return self.score(game, game.active_player)
            return - self.score(game, game.inactive_player)
        #let's give a lower bound to the possible score
        best_score = float("-inf")
        child_nodes = game.get_legal_moves()
        #sort child_nodes before alpha-beta prunning
        #sorted(child_nodes, key = lambda x : self.score(game.forecast_move(x), game.inactive_player), reverse=True)
        #look at the child nodes
        for move in child_nodes:
            #this is the recurive part with depth decreasing gradually:
            #the higher score our opponent give to a node, the lower score we give to it
            #in each recursion, we switch the perspective from player to his/her opponent
            #so the role of of alpha and beta have to change accordingly
            next_game = game.forecast_move(move)
            test_score = - self.evaluate_alphabeta(next_game, depth-1, -beta, -alpha, not current_player)
            #improve our best_score whenever possible
            best_score = max(best_score, test_score)
            #update the record of alpha whenever necessary
            #value of beta will be updated automatically in recursive call of this function
            alpha = max(best_score, alpha)
            #prunning
            if alpha >= beta:
                break
        return best_score
    