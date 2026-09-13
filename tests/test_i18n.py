import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PyQt6.QtWidgets import QApplication, QLabel
from optics2d.desktop import OpticsDesktop
from optics2d.i18n import Translator, translator
from optics2d.visualization import KIND_LABELS


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.translator = Translator()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'language.json'

    def write(self, messages, code='de'):
        self.path.write_text(json.dumps({'code': code, 'name': 'Deutsch',
                                        'messages': messages}), encoding='utf-8')

    def test_default_and_complete_builtins(self):
        self.assertEqual(self.translator.code, 'en')
        self.assertEqual(self.translator.catalogs['en']['messages'].keys(),
                         self.translator.catalogs['ru']['messages'].keys())
        self.translator.set_language('ru')
        self.assertEqual(self.translator.translate('Horn'), 'Рупор')
        self.assertEqual(self.translator.translate('Previous object: {0}', 'M1'),
                         'Предыдущий объект: M1')

    def test_partial_catalog_and_placeholder_reordering(self):
        self.write({'Horn': 'Horn DE', 'in:{0}  out:{1}': 'aus:{1} ein:{0}'})
        self.translator.set_language(self.translator.load(self.path))
        self.assertEqual(self.translator.translate('Horn'), 'Horn DE')
        self.assertEqual(self.translator.translate('Undo'), 'Undo')
        self.assertEqual(self.translator.translate('in:{0}  out:{1}', 2, 3), 'aus:3 ein:2')

    def test_bad_catalogs_do_not_replace_active_language(self):
        for messages in ({'Previous object: {0}': 'Missing value'},
                         {'Previous object: {0}': '{0.__class__}'},
                         {'Horn': 123}, {'Unknown key': 'text'},
                         {'Previous object: {0}': '{0:bad}'}):
            self.write(messages)
            with self.assertRaises(ValueError):
                self.translator.load(self.path)
            self.assertNotIn('de', self.translator.catalogs)
            self.assertEqual(self.translator.code, 'en')


class LanguageSwitchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        translator.set_language('en')
        self.window = OpticsDesktop()
        self.addCleanup(self.window.close)
        self.addCleanup(translator.set_language, 'en')

    def test_switch_preserves_model_history_selection_and_pending_fields(self):
        w = self.window
        w.add_element('horn')
        w.field_editors['x'].setText('123.456')
        w.frequency_table.item(0, 1).setText('299')
        w.max_path_editor.setText('1234')
        state = w._history_state()
        history = list(w._undo_stack)
        for code, label in [('ru', 'Рупор'), ('en', 'Horn')]:
            w._change_language(code)
            self.app.processEvents()
            self.assertEqual(KIND_LABELS['horn'], label)
            self.assertEqual(w._history_state(), state)
            self.assertEqual(w._undo_stack, history)
            self.assertEqual(w.field_editors['x'].text(), '123.456')
            self.assertEqual(w.max_path_editor.text(), '1234')
            self.assertEqual(w.frequency_table.item(0, 1).text(), '299')
        w.undo()
        self.assertEqual(len(w.system.elements), len(state['system']['elements']) - 1)

    def test_english_interface_and_all_object_forms(self):
        w = self.window
        for kind in KIND_LABELS:
            w.add_element(kind)
            for code in ('ru', 'en'):
                w._change_language(code)
                self.app.processEvents()
            for label in w.centralWidget().findChildren(QLabel):
                self.assertFalse(any('\u0400' <= ch <= '\u04ff' for ch in label.text()), label.text())

    def test_load_file_through_interface_and_reject_invalid_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'de.json'
            path.write_text(json.dumps({'code': 'de', 'name': 'Deutsch',
                                        'messages': {'Undo': 'Rückgängig'}}), encoding='utf-8')
            with patch('optics2d.desktop.QFileDialog.getOpenFileName', return_value=(str(path), '')):
                self.window.load_language_file()
                self.app.processEvents()
                self.assertEqual(self.window.language_combo.currentData(), 'de')
                self.assertEqual(self.window.undo_action.text(), 'Rückgängig')
                self.assertEqual(self.window.redo_action.text(), 'Redo')
                path.write_text('{broken json', encoding='utf-8')
                with patch('optics2d.desktop.QMessageBox.critical') as error:
                    self.window.load_language_file()
                    error.assert_called_once()
                self.assertEqual(translator.code, 'de')
            translator.set_language('en')
            del translator.catalogs['de']


if __name__ == '__main__':
    unittest.main()
