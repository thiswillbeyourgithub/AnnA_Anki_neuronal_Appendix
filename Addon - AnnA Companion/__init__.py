# Copyright 2016-2021 Alex Yatskov
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import inspect
import json

from PyQt5 import QtCore
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMessageBox

import anki
import anki.exporting
import anki.storage
import aqt
from anki.cards import Card

from anki.notes import Note

try:
    from anki.rsbackend import NotFoundError
except:
    NotFoundError = Exception

from . import web, util


#
# AnkiConnect
#

class AnkiConnect:
    _anki21_version = int(aqt.appVersion.split('.')[-1])

    def __init__(self):
        self.log = None
        logPath = util.setting('apiLogPath')
        if logPath is not None:
            self.log = open(logPath, 'w')

        try:
            self.server = web.WebServer(self.handler)
            self.server.listen()

            self.timer = QTimer()
            self.timer.timeout.connect(self.advance)
            self.timer.start(util.setting('apiPollInterval'))
        except:
            QMessageBox.critical(
                self.window(),
                'AnkiConnect',
                'Failed to listen on port {}.\nMake sure it is available and is not in use.'.format(util.setting('webBindPort'))
            )

    def save_model(self, models, ankiModel):
        if self._anki21_version < 45:
            models.save(ankiModel, True)
            models.flush()
        else:
            models.update_dict(ankiModel)


    def logEvent(self, name, data):
        if self.log is not None:
            self.log.write('[{}]\n'.format(name))
            json.dump(data, self.log, indent=4, sort_keys=True)
            self.log.write('\n\n')
            self.log.flush()


    def advance(self):
        self.server.advance()


    def handler(self, request):
        self.logEvent('request', request)

        name = request.get('action', '')
        version = request.get('version', 4)
        params = request.get('params', {})
        key = request.get('key')
        reply = {'result': None, 'error': None}

        try:
            if key != util.setting('apiKey') and name != 'requestPermission':
                raise Exception('valid api key must be provided')

            method = None

            for methodName, methodInst in inspect.getmembers(self, predicate=inspect.ismethod):
                apiVersionLast = 0
                apiNameLast = None

                if getattr(methodInst, 'api', False):
                    for apiVersion, apiName in getattr(methodInst, 'versions', []):
                        if apiVersionLast < apiVersion <= version:
                            apiVersionLast = apiVersion
                            apiNameLast = apiName

                    if apiNameLast is None and apiVersionLast == 0:
                        apiNameLast = methodName

                    if apiNameLast is not None and apiNameLast == name:
                        method = methodInst
                        break

            if method is None:
                raise Exception('unsupported action. You addon version is probably outdated.')
            else:
                reply['result'] = methodInst(**params)

            if version <= 4:
                reply = reply['result']

        except Exception as e:
            reply['error'] = str(e)

        self.logEvent('reply', reply)
        return reply


    def window(self):
        return aqt.mw


    def reviewer(self):
        reviewer = self.window().reviewer
        if reviewer is None:
            raise Exception('reviewer is not available')

        return reviewer


    def collection(self):
        collection = self.window().col
        if collection is None:
            raise Exception('collection is not available')

        return collection


    def decks(self):
        decks = self.collection().decks
        if decks is None:
            raise Exception('decks are not available')

        return decks


    def scheduler(self):
        scheduler = self.collection().sched
        if scheduler is None:
            raise Exception('scheduler is not available')

        return scheduler


    def database(self):
        database = self.collection().db
        if database is None:
            raise Exception('database is not available')

        return database


    def media(self):
        media = self.collection().media
        if media is None:
            raise Exception('media is not available')

        return media


    def startEditing(self):
        self.window().requireReset()


    def stopEditing(self):
        if self.collection() is not None:
            self.window().maybeReset()


    def createNote(self, note):
        collection = self.collection()

        model = collection.models.byName(note['modelName'])
        if model is None:
            raise Exception('model was not found: {}'.format(note['modelName']))

        deck = collection.decks.byName(note['deckName'])
        if deck is None:
            raise Exception('deck was not found: {}'.format(note['deckName']))

        ankiNote = anki.notes.Note(collection, model)
        ankiNote.model()['did'] = deck['id']
        if 'tags' in note:
            ankiNote.tags = note['tags']

        for name, value in note['fields'].items():
            for ankiName in ankiNote.keys():
                if name.lower() == ankiName.lower():
                    ankiNote[ankiName] = value
                    break

        allowDuplicate = False
        duplicateScope = None
        duplicateScopeDeckName = None
        duplicateScopeCheckChildren = False
        duplicateScopeCheckAllModels = False

        if 'options' in note:
            options = note['options']
            if 'allowDuplicate' in options:
                allowDuplicate = options['allowDuplicate']
                if type(allowDuplicate) is not bool:
                    raise Exception('option parameter "allowDuplicate" must be boolean')
            if 'duplicateScope' in options:
                duplicateScope = options['duplicateScope']
            if 'duplicateScopeOptions' in options:
                duplicateScopeOptions = options['duplicateScopeOptions']
                if 'deckName' in duplicateScopeOptions:
                    duplicateScopeDeckName = duplicateScopeOptions['deckName']
                if 'checkChildren' in duplicateScopeOptions:
                    duplicateScopeCheckChildren = duplicateScopeOptions['checkChildren']
                    if type(duplicateScopeCheckChildren) is not bool:
                        raise Exception('option parameter "duplicateScopeOptions.checkChildren" must be boolean')
                if 'checkAllModels' in duplicateScopeOptions:
                    duplicateScopeCheckAllModels = duplicateScopeOptions['checkAllModels']
                    if type(duplicateScopeCheckAllModels) is not bool:
                        raise Exception('option parameter "duplicateScopeOptions.checkAllModels" must be boolean')

        duplicateOrEmpty = self.isNoteDuplicateOrEmptyInScope(
            ankiNote,
            deck,
            collection,
            duplicateScope,
            duplicateScopeDeckName,
            duplicateScopeCheckChildren,
            duplicateScopeCheckAllModels
        )

        if duplicateOrEmpty == 1:
            raise Exception('cannot create note because it is empty')
        elif duplicateOrEmpty == 2:
            if allowDuplicate:
                return ankiNote
            raise Exception('cannot create note because it is a duplicate')
        elif duplicateOrEmpty == 0:
            return ankiNote
        else:
            raise Exception('cannot create note for unknown reason')


    def isNoteDuplicateOrEmptyInScope(
        self,
        note,
        deck,
        collection,
        duplicateScope,
        duplicateScopeDeckName,
        duplicateScopeCheckChildren,
        duplicateScopeCheckAllModels
    ):
        # Returns: 1 if first is empty, 2 if first is a duplicate, 0 otherwise.

        # note.dupeOrEmpty returns if a note is a global duplicate with the specific model.
        # This is used as the default check, and the rest of this function is manually
        # checking if the note is a duplicate with additional options.
        if duplicateScope != 'deck' and not duplicateScopeCheckAllModels:
            return note.dupeOrEmpty() or 0

        # Primary field for uniqueness
        val = note.fields[0]
        if not val.strip():
            return 1
        csum = anki.utils.fieldChecksum(val)

        # Create dictionary of deck ids
        dids = None
        if duplicateScope == 'deck':
            did = deck['id']
            if duplicateScopeDeckName is not None:
                deck2 = collection.decks.byName(duplicateScopeDeckName)
                if deck2 is None:
                    # Invalid deck, so cannot be duplicate
                    return 0
                did = deck2['id']

            dids = {did: True}
            if duplicateScopeCheckChildren:
                for kv in collection.decks.children(did):
                    dids[kv[1]] = True

        # Build query
        query = 'select id from notes where csum=?'
        queryArgs = [csum]
        if note.id:
            query += ' and id!=?'
            queryArgs.append(note.id)
        if not duplicateScopeCheckAllModels:
            query += ' and mid=?'
            queryArgs.append(note.mid)

        # Search
        for noteId in note.col.db.list(query, *queryArgs):
            if dids is None:
                # Duplicate note exists in the collection
                return 2
            # Validate that a card exists in one of the specified decks
            for cardDeckId in note.col.db.list('select did from cards where nid=?', noteId):
                if cardDeckId in dids:
                    return 2

        # Not a duplicate
        return 0

    def getCard(self, card_id: int) -> Card:
        try:
            return self.collection().getCard(card_id)
        except NotFoundError:
            raise NotFoundError('Card was not found: {}'.format(card_id))

    def getNote(self, note_id: int) -> Note:
        try:
            return self.collection().getNote(note_id)
        except NotFoundError:
            raise NotFoundError('Note was not found: {}'.format(note_id))

    @util.api()
    def version(self):
        return util.setting('apiVersion')

    @util.api()
    def requestPermission(self, origin, allowed):
        if allowed:
            return {
                "permission": "granted",
                "requireApikey": bool(util.setting('apiKey')),
                "version": util.setting('apiVersion')
            }

        if origin in util.setting('ignoreOriginList') :
            return {
                "permission": "denied",
            }

        msg = QMessageBox(None)
        msg.setWindowTitle("A website want to access to Anki")
        msg.setText(origin + " request permission to use Anki through AnkiConnect.\nDo you want to give it access ?")
        msg.setInformativeText("By giving permission, the website will be able to do actions on anki, including destructives actions like deck deletion.")
        msg.setWindowIcon(self.window().windowIcon())
        msg.setIcon(QMessageBox.Question)
        msg.setStandardButtons(QMessageBox.Yes|QMessageBox.Ignore|QMessageBox.No)
        msg.setDefaultButton(QMessageBox.No)
        msg.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        pressedButton = msg.exec_()

        if pressedButton == QMessageBox.Yes:
            config = aqt.mw.addonManager.getConfig(__name__)
            config["webCorsOriginList"] = util.setting('webCorsOriginList')
            config["webCorsOriginList"].append(origin)
            aqt.mw.addonManager.writeConfig(__name__, config)

        if pressedButton == QMessageBox.Ignore:
            config = aqt.mw.addonManager.getConfig(__name__)
            config["ignoreOriginList"] = util.setting('ignoreOriginList')
            config["ignoreOriginList"].append(origin)
            aqt.mw.addonManager.writeConfig(__name__, config)

        if pressedButton == QMessageBox.Yes:
            results = {
                "permission": "granted",
                "requireApikey": bool(util.setting('apiKey')),
                "version": util.setting('apiVersion')
            }
        else :
            results = {
                "permission": "denied",
            }
        return results

    @util.api()
    def deckNames(self):
        return self.decks().allNames()

    @util.api()
    def getDeckConfig(self, deck):
        if deck not in self.deckNames():
            return False

        collection = self.collection()
        did = collection.decks.id(deck)
        return collection.decks.confForDid(did)

    @util.api()
    def findCards(self, query=None):
        if query is None:
            return []

        return list(map(int, self.collection().findCards(query)))

    @util.api()
    def createFilteredDeck(self, newDeckName='New filtered deck', searchQuery='', gatherCount=50, reschedule=True, sortOrder=0, createEmpty=False):
        # first checks if the deck name is not already taken
        deckList = self.decks().allNames()
        newDeckName = str(newDeckName)
        if newDeckName in deckList:
            return False  # deckname already taken

        did = self.collection().decks.newDyn(newDeckName)
        d = self.collection().decks.current()
        d['terms'] = [[str(searchQuery), int(gatherCount), sortOrder]]
        d['resched'] = reschedule
        self.collection().decks.save(d)
        if createEmpty is False:
            aqt.mw.col.sched.rebuild_filtered_deck(did)
            aqt.mw.reset()
        return did

    @util.api()
    def setDueOrderOfFiltered(self, cards):
        result = []
        order = -100_000
        for card in cards:
            order += 1
            try:
                ankiCard = self.getCard(card)
                ankiCard.due = order
                ankiCard.flush()
                result.append([True])
            except Exception as e:
                result.append([False, e])
        return result

    @util.api()
    def bury(self, cards, bury=True):
        for card in cards:
            if self.buried(card) == bury:
                cards.remove(card)

        scheduler = self.scheduler()
        self.startEditing()
        if bury:
            scheduler.buryCards(cards)
        else:
            scheduler.unburyCards(cards)
        self.stopEditing()

        return True

    @util.api()
    def unbury(self, cards):
        self.bury(cards, False)

    @util.api()
    def buried(self, card):
        card = self.getCard(card)
        if card.queue in [-3, -2]:
            return card.queue
        else:
            return False

    @util.api()
    def areBuried(self, cards):
        buried = []
        for card in cards:
            try:
                buried.append(self.buried(card))
            except NotFoundError:
                buried.append(None)
        return buried

    @util.api()
    def cardsInfo(self, cards):
        result = []
        for cid in cards:
            try:
                card = self.getCard(cid)
                model = card.note_type()
                note = card.note()
                fields = {}

                for info in model['flds']:
                    fields[info['name']] = {'value': note.fields[info['ord']],
                                            'order': info['ord']}

                result.append({
                    'cardId': card.id,
                    'fields': fields,
                    'modelName': model['name'],
                    'tags': note.tags,
                    'interval': card.ivl,
                    'note': card.nid,
                    'type': card.type,
                    'queue': card.queue,
                    'due': card.due,
                    'odue': card.odue,
                    'left': card.left,
                })
            except NotFoundError:
                # Anki will give a NotFoundError if the card ID does not exist.
                # Best behavior is probably to add an 'empty card' to the
                # returned result, so that the items of the input and return
                # lists correspond.
                result.append({})
        return result

    @util.api()
    def getCollectionCreationTime(self):
        try:
            return aqt.mw.col.crt
        except Exception:
            return False

    @util.api()
    def addTags(self, notes, tags, add=True):
        self.startEditing()
        self.collection().tags.bulkAdd(notes, tags, add)
        self.stopEditing()

    @util.api()
    def cardsToNotes(self, cards):
        return self.collection().db.list('select distinct nid from cards where id in ' + anki.utils.ids2str(cards))

    @util.api()
    def update_KNN_field(self, notes=None, new_field_value=None):
        self.startEditing()
        for i in range(len(notes)):
            ankiNote = self.getNote(notes[i])
            if ankiNote["KNN_neighbours"] != new_field_value[i]:
                ankiNote["KNN_neighbours"] = new_field_value[i]
                ankiNote.flush()
        self.collection().autosave()
        self.stopEditing()


ac = AnkiConnect()
